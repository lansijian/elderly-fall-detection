import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils import weight_norm
import pywt


class WaveletEmbedding(nn.Module):
    """小波变换嵌入层，用于时间序列的多尺度特征提取"""
    
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                kernel_size=None):
        super().__init__()

        self.swt = swt  # 是否使用小波变换
        self.d_channel = d_channel  # 通道数
        self.m = m  # 分解级别

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)
        
            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        # 检查输入通道数是否与模型通道数匹配
        batch_size, n_channels, seq_len = x.shape
        
        if n_channels != self.d_channel:
            print(f"WaveletEmbedding: 调整通道数从 {self.d_channel} 到 {n_channels}")
            self.d_channel = n_channels
            
            # 重新初始化滤波器以匹配新的通道数
            if hasattr(self, 'wavelet'):
                if self.swt:
                    h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32, device=x.device)
                    h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32, device=x.device)
                else:
                    h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32, device=x.device)
                    h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32, device=x.device)
                
                # 创建新的参数并替换旧参数
                new_h0 = nn.Parameter(torch.tile(h0[None, None, :], [n_channels, 1, 1]), 
                                    requires_grad=self.h0.requires_grad).to(x.device)
                new_h1 = nn.Parameter(torch.tile(h1[None, None, :], [n_channels, 1, 1]), 
                                    requires_grad=self.h1.requires_grad).to(x.device)
                self.h0 = new_h0
                self.h1 = new_h1
            else:
                # 如果没有wavelet属性，使用xavier初始化
                new_h0 = nn.Parameter(torch.Tensor(n_channels, 1, self.kernel_size), 
                                    requires_grad=self.h0.requires_grad).to(x.device)
                new_h1 = nn.Parameter(torch.Tensor(n_channels, 1, self.kernel_size), 
                                    requires_grad=self.h1.requires_grad).to(x.device)
                nn.init.xavier_uniform_(new_h0)
                nn.init.xavier_uniform_(new_h1)
                
                with torch.no_grad():
                    new_h0.data = new_h0.data / torch.norm(new_h0.data, dim=-1, keepdim=True)
                    new_h1.data = new_h1.data / torch.norm(new_h1.data, dim=-1, keepdim=True)
                
                self.h0 = new_h0
                self.h1 = new_h1
        
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        """
        小波分解
        
        参数:
            x: 输入数据 [batch_size, channels, length]
            h0, h1: 小波滤波器
            depth: 分解级别
            kernel_size: 核大小
            
        返回:
            小波系数
        """
        # 确保h0和h1的通道数与x匹配
        batch_size, n_channels, seq_len = x.shape
        
        # 如果滤波器的通道数与输入不匹配，调整滤波器
        if h0.shape[0] != n_channels:
            print(f"调整小波滤波器通道数: 从 {h0.shape[0]} 到 {n_channels}")
            # 创建新的滤波器，复制到所有通道
            new_h0 = torch.zeros(n_channels, 1, kernel_size, device=x.device)
            new_h1 = torch.zeros(n_channels, 1, kernel_size, device=x.device)
            
            # 填充滤波器权重（针对每个通道使用相同的滤波器）
            for i in range(n_channels):
                idx = i % h0.shape[0]  # 循环使用原始滤波器
                new_h0[i, 0, :] = h0[idx, 0, :]
                new_h1[i, 0, :] = h1[idx, 0, :]
                
            h0, h1 = new_h0, new_h1
        
        approx_coeffs = x
        coeffs = []
        dilation = 1
        
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            
            # 尝试使用groups=n_channels进行卷积
            try:
                detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=n_channels)
                approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=n_channels)
            except RuntimeError as e:
                # 如果groups参数不匹配，尝试使用groups=1
                print(f"卷积错误: {e}")
                print("尝试使用groups=1进行卷积...")
                try:
                    detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=1)
                    approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=1)
                except RuntimeError as e2:
                    # 如果仍然失败，尝试使用最小公约数作为groups
                    print(f"卷积错误2: {e2}")
                    try:
                        import math
                        gcd = math.gcd(n_channels, h1.shape[0])
                        print(f"尝试使用groups={gcd}进行卷积...")
                        detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=gcd)
                        approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=gcd)
                    except:
                        # 如果所有尝试都失败，使用循环逐通道处理
                        print("尝试逐通道处理...")
                        detail_coeff = torch.zeros_like(approx_coeffs)
                        new_approx = torch.zeros_like(approx_coeffs)
                        for i in range(n_channels):
                            ch_input = approx_coeffs_pad[:, i:i+1, :]
                            ch_h0 = h0[i % h0.shape[0]:i % h0.shape[0]+1, :, :]
                            ch_h1 = h1[i % h1.shape[0]:i % h1.shape[0]+1, :, :]
                            detail_coeff[:, i:i+1, :] = F.conv1d(ch_input, ch_h1, dilation=dilation)
                            new_approx[:, i:i+1, :] = F.conv1d(ch_input, ch_h0, dilation=dilation)
                        approx_coeffs = new_approx
            
            coeffs.append(detail_coeff)
            dilation *= 2
            
        coeffs.append(approx_coeffs)
        
        # 按照分解级别排序并堆叠
        result = torch.stack(list(reversed(coeffs)), -2)
        return result

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        """
        小波重构
        
        参数:
            coeffs: 小波系数 [batch_size, channels, length] 或 [batch_size, channels, levels, length]
            g0, g1: 小波重构滤波器
            m: 分解级别
            kernel_size: 核大小
            
        返回:
            重构后的数据
        """
        # 检查输入维度
        if len(coeffs.shape) == 3:
            # 如果是3维输入，我们将其视为已经是最终的近似系数
            # 直接返回，无需重构
            return coeffs
        
        batch_size, n_channels, _, seq_len = coeffs.shape
        
        # 确保g0和g1的通道数与输入匹配
        if g0.shape[0] != n_channels:
            print(f"调整重构滤波器通道数: 从 {g0.shape[0]} 到 {n_channels}")
            # 创建新的滤波器，复制到所有通道
            new_g0 = torch.zeros(n_channels, 1, kernel_size, device=coeffs.device)
            new_g1 = torch.zeros(n_channels, 1, kernel_size, device=coeffs.device)
            
            # 填充滤波器权重（针对每个通道使用相同的滤波器）
            for i in range(n_channels):
                idx = i % g0.shape[0]  # 循环使用原始滤波器
                new_g0[i, 0, :] = g0[idx, 0, :]
                new_g1[i, 0, :] = g1[idx, 0, :]
                
            g0, g1 = new_g0, new_g1
        
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            # 尝试使用groups=n_channels进行卷积
            try:
                y = F.conv1d(approx_coeff_pad, g0, groups=n_channels, dilation=dilation) + \
                    F.conv1d(detail_coeff_pad, g1, groups=n_channels, dilation=dilation)
            except RuntimeError as e:
                # 如果groups参数不匹配，尝试使用groups=1
                print(f"重构卷积错误: {e}")
                print("尝试使用groups=1进行重构卷积...")
                try:
                    y = F.conv1d(approx_coeff_pad, g0, groups=1, dilation=dilation) + \
                        F.conv1d(detail_coeff_pad, g1, groups=1, dilation=dilation)
                except RuntimeError as e2:
                    # 如果仍然失败，尝试使用最小公约数作为groups
                    print(f"重构卷积错误2: {e2}")
                    try:
                        import math
                        gcd = math.gcd(n_channels, g0.shape[0])
                        print(f"尝试使用groups={gcd}进行重构卷积...")
                        y = F.conv1d(approx_coeff_pad, g0, groups=gcd, dilation=dilation) + \
                            F.conv1d(detail_coeff_pad, g1, groups=gcd, dilation=dilation)
                    except:
                        # 如果所有尝试都失败，使用循环逐通道处理
                        print("尝试逐通道进行重构...")
                        y = torch.zeros_like(approx_coeff)
                        for i in range(n_channels):
                            ch_approx = approx_coeff_pad[:, i:i+1, :]
                            ch_detail = detail_coeff_pad[:, i:i+1, :]
                            ch_g0 = g0[i % g0.shape[0]:i % g0.shape[0]+1, :, :]
                            ch_g1 = g1[i % g1.shape[0]:i % g1.shape[0]+1, :, :]
                            y[:, i:i+1, :] = F.conv1d(ch_approx, ch_g0, dilation=dilation) + \
                                              F.conv1d(ch_detail, ch_g1, dilation=dilation)
                
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff


class GeomAttention(nn.Module):
    """几何注意力机制，结合点积和楔形积信息"""
    
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, 
                output_attention=False, alpha=1.):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.alpha = alpha  # 控制点积和楔形积的平衡系数

    def forward(self, queries, keys, values, attn_mask=None):
        # 检查输入维度
        if len(queries.shape) == 3:
            # 如果是3维输入 [batch_size, seq_len, embed_dim]
            # 重塑为4维 [batch_size, seq_len, 1, embed_dim]
            B, L, E = queries.shape
            H = 1  # 只使用一个注意力头
            queries = queries.unsqueeze(2)  # [B, L, 1, E]
            keys = keys.unsqueeze(2)        # [B, L, 1, E]
            values = values.unsqueeze(2)    # [B, L, 1, E]
        else:
            # 如果已经是4维输入 [batch_size, seq_len, n_heads, embed_dim]
            B, L, H, E = queries.shape
        
        # 获取序列长度S
        S = keys.shape[1]
        
        scale = self.scale or 1. / math.sqrt(E)

        # 计算传统的点积注意力
        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 计算几何楔形积
        queries_norm2 = torch.sum(queries**2, dim=-1)
        keys_norm2 = torch.sum(keys**2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)        # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2   # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)  # 确保非负
        
        # 几何注意力分数，结合点积和楔形积
        scores = self.alpha * dot_product * scale - (1 - self.alpha) * wedge_norm2 * scale

        # 添加掩码（如果有）
        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill_(attn_mask, -np.inf)
        
        # 归一化注意力权重
        attention = torch.softmax(scores, dim=-1)  # [B, H, L, S]
        attention = self.dropout(attention)
        
        # 计算上下文向量 [B, H, L, S] x [B, S, H, E] -> [B, H, L, E]
        # 首先调整值的形状
        values_t = values.permute(0, 1, 3, 2)  # [B, S, E, H]
        values_t = values_t.permute(0, 3, 1, 2)  # [B, H, S, E]
        
        # 进行矩阵乘法
        out = torch.matmul(attention, values_t)  # [B, H, L, E]
        out = out.permute(0, 2, 1, 3)  # [B, L, H, E]
        
        # 如果原始输入是3维，则返回3维输出
        if len(queries.shape) == 4 and H == 1:
            out = out.squeeze(2)  # 去除头维度 [B, L, E]
        
        if self.output_attention:
            return out, attention
        else:
            return out, None


class GeomAttentionLayer(nn.Module):
    """几何注意力层，封装几何注意力机制和周围的处理"""
    
    def __init__(self, attention, d_model, requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5):
        super(GeomAttentionLayer, self).__init__()

        # 确保d_channel有值，默认使用d_model
        self.d_channel = d_model if d_channel is None else d_channel
        self.d_model = d_model      # 模型维度
        self.inner_attention = attention
        
        # 小波变换嵌入 - 使用全部通道
        self.swt = WaveletEmbedding(
            d_channel=self.d_channel, 
            swt=True, 
            requires_grad=requires_grad, 
            wv=wv, 
            m=m, 
            kernel_size=kernel_size
        )
        
        # 计算小波变换后的维度，考虑到分解级别m
        self.wave_dim = self.d_channel * (m + 1)  # m个细节系数 + 1个近似系数
        
        # 查询、键、值的投影
        self.query_projection = nn.Sequential(
            nn.Linear(self.wave_dim, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(self.wave_dim, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(self.wave_dim, d_model),
            nn.Dropout(geomattn_dropout)
        )
        
        # 输出投影 - 确保小波重构使用相同数量的通道
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, self.d_channel),  # 从d_model投影回d_channel
            WaveletEmbedding(
                d_channel=self.d_channel, 
                swt=False, 
                requires_grad=requires_grad, 
                wv=wv, 
                m=m, 
                kernel_size=kernel_size
            ),
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch_size, seq_len, emb_dim = queries.shape
        
        # 确保输入维度与d_channel匹配
        if emb_dim != self.d_channel:
            # 如果不匹配，重新初始化投影层以适应新的维度
            print(f"输入维度 ({emb_dim}) 与d_channel ({self.d_channel}) 不匹配，调整维度")
            self.d_channel = emb_dim
            self.wave_dim = self.d_channel * (self.swt.m + 1)
            
            # 重新创建投影层
            self.query_projection = nn.Sequential(
                nn.Linear(self.wave_dim, self.d_model),
                nn.Dropout(self.query_projection[1].p)
            ).to(queries.device)
            self.key_projection = nn.Sequential(
                nn.Linear(self.wave_dim, self.d_model),
                nn.Dropout(self.key_projection[1].p)
            ).to(queries.device)
            self.value_projection = nn.Sequential(
                nn.Linear(self.wave_dim, self.d_model),
                nn.Dropout(self.value_projection[1].p)
            ).to(queries.device)
            self.out_projection = nn.Sequential(
                nn.Linear(self.d_model, self.d_channel),
                WaveletEmbedding(
                    d_channel=self.d_channel, 
                    swt=False, 
                    requires_grad=self.swt.h0.requires_grad, 
                    wv=self.swt.wavelet.name if hasattr(self.swt, 'wavelet') else 'db2', 
                    m=self.swt.m, 
                    kernel_size=self.swt.kernel_size
                )
            ).to(queries.device)
        
        # 对查询、键、值应用小波变换
        queries = self.swt(queries.transpose(1, 2))  # [B, D, L] -> [B, D, m+1, L]
        keys = self.swt(keys.transpose(1, 2))        # [B, D, L] -> [B, D, m+1, L]
        values = self.swt(values.transpose(1, 2))    # [B, D, L] -> [B, D, m+1, L]
        
        # 转置回 [B, L, m+1, D] 以便进行线性投影
        queries = queries.permute(0, 3, 2, 1)  # [B, L, m+1, D]
        keys = keys.permute(0, 3, 2, 1)        # [B, L, m+1, D]
        values = values.permute(0, 3, 2, 1)    # [B, L, m+1, D]
        
        # 获取形状参数
        m_plus_one = queries.shape[2]  # m+1
        
        # 重塑为二维张量以便线性投影
        queries = queries.reshape(batch_size * seq_len, m_plus_one * self.d_channel)
        keys = keys.reshape(batch_size * seq_len, m_plus_one * self.d_channel)
        values = values.reshape(batch_size * seq_len, m_plus_one * self.d_channel)
        
        # 投影到模型维度
        queries = self.query_projection(queries)  # [B*L, D']
        keys = self.key_projection(keys)          # [B*L, D']
        values = self.value_projection(values)    # [B*L, D']
        
        # 重塑回三维张量 [B, L, D']
        queries = queries.reshape(batch_size, seq_len, self.d_model)
        keys = keys.reshape(batch_size, seq_len, self.d_model)
        values = values.reshape(batch_size, seq_len, self.d_model)
        
        # 应用几何注意力
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        # 如果attn是None，创建一个标量张量替代
        if attn is None:
            attn = torch.tensor(0.0, device=out.device)
        
        # 输出投影和重构
        out = out.reshape(batch_size * seq_len, self.d_model)
        out = self.out_projection[0](out)  # 线性投影 [B*L, D_channel]
        out = out.reshape(batch_size, seq_len, self.d_channel)
        out = out.transpose(1, 2)  # [B, D_channel, L]
        out = self.out_projection[1](out)  # 小波重构
        out = out.transpose(1, 2)  # [B, L, D_channel]
        
        return out, attn


class EncoderLayer(nn.Module):
    """编码器层，包含几何注意力和前馈网络"""
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", dec_in=6):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 多头自注意力机制
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        
        # 如果attn是None，创建一个标量张量替代
        if attn is None:
            attn = torch.tensor(0.0, device=x.device)
            
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        # 前馈网络（使用1D卷积实现）
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 残差连接和层归一化
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """编码器，包含多个编码器层"""
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class SimpleDataEmbedding(nn.Module):
    """简单的数据嵌入层，将传感器特征转换为模型维度"""
    
    def __init__(self, n_features, d_model, dropout=0.1):
        """
        初始化数据嵌入层
        
        参数:
            n_features (int): 输入特征维度
            d_model (int): 模型维度
            dropout (float): Dropout比例
        """
        super(SimpleDataEmbedding, self).__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # 使用1D卷积进行特征映射
        self.feature_projection = nn.Sequential(
            nn.Conv1d(n_features, d_model // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=1)
        )
        
        # 添加位置编码
        self.position_encoding = nn.Parameter(
            torch.zeros(1, 10000, d_model),  # 支持最长10000的序列
            requires_grad=False
        )
        
        # 使用dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化位置编码
        position = torch.arange(10000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pos_encoding = torch.zeros(1, 10000, self.d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.position_encoding.data = pos_encoding
    
    def forward(self, x, x_mark=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [batch_size, n_features, seq_len]
            x_mark (Tensor): 时间编码 (未使用)
            
        返回:
            Tensor: 嵌入后的张量 [batch_size, seq_len, d_model]
        """
        # 检查输入维度
        batch_size, n_features, seq_len = x.shape
        
        # 打印调试信息
        # print(f"SimpleDataEmbedding input shape: {x.shape}")
        
        # 确保特征维度匹配
        if n_features != self.n_features:
            print(f"警告: 输入特征维度 ({n_features}) 与预期特征维度 ({self.n_features}) 不匹配")
            # 如果输入特征维度不匹配，动态调整特征映射层
            self.feature_projection = nn.Sequential(
                nn.Conv1d(n_features, self.d_model // 2, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.d_model // 2, self.d_model, kernel_size=1)
            ).to(x.device)
            self.n_features = n_features
        
        # 应用特征映射 [batch_size, n_features, seq_len] -> [batch_size, d_model, seq_len]
        x = self.feature_projection(x)
        
        # 转置为 [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # 添加位置编码
        x = x + self.position_encoding[:, :seq_len, :]
        
        # 应用dropout
        x = self.dropout(x)
        
        # print(f"SimpleDataEmbedding output shape: {x.shape}")
        return x


class SimpleTM_FallDetector(nn.Module):
    """基于SimpleTM的跌倒检测模型"""
    
    def __init__(self, configs):
        super(SimpleTM_FallDetector, self).__init__()
        self.task_name = 'classification'  # 跌倒检测是分类任务
        self.seq_len = configs.seq_len
        self.pred_len = 1  # 分类任务只需要输出一个标签
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.geomattn_dropout = configs.geomattn_dropout
        self.alpha = configs.alpha
        self.kernel_size = configs.kernel_size
        self.num_classes = configs.num_classes  # 类别数，跌倒检测通常是二分类
        
        # 数据嵌入层 - 使用简化的嵌入层，直接指定输入特征数量
        self.enc_embedding = SimpleDataEmbedding(
            n_features=configs.dec_in,  # 输入特征数
            d_model=configs.d_model,    # 模型维度
            dropout=configs.dropout
        )

        # 编码器层
        encoder = Encoder(
            [  
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention, alpha=self.alpha
                        ),
                        configs.d_model, 
                        requires_grad=configs.requires_grad, 
                        wv=configs.wv, 
                        m=configs.m, 
                        d_channel=configs.d_model,  # 使用模型维度作为通道数 
                        kernel_size=self.kernel_size, 
                        geomattn_dropout=self.geomattn_dropout
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    dec_in=configs.dec_in
                ) for l in range(configs.e_layers) 
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.encoder = encoder

        # 分类头：全连接层，用于将特征映射到类别概率
        self.classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, self.num_classes)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 添加调试信息
        # print(f"Input shape: x_enc={x_enc.shape}")
        
        if self.use_norm:
            # 数据归一化
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # 数据嵌入
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 
        # print(f"After embedding shape: enc_out={enc_out.shape}")

        # 通过SimpleTM编码器
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # print(f"After encoder shape: enc_out={enc_out.shape}")
        
        # 对序列特征进行全局平均池化，得到固定长度的特征向量
        # 形状从 [batch_size, seq_len, d_model] 变为 [batch_size, d_model]
        enc_out = torch.mean(enc_out, dim=1)
        # print(f"After pooling shape: enc_out={enc_out.shape}")
        
        # 通过分类头生成类别预测
        output = self.classifier(enc_out)
        # print(f"Output shape: output={output.shape}")
        
        # 计算L1正则化项
        l1_reg = 0.0
        for attn in attns:
            l1_reg += torch.mean(torch.abs(attn))
        
        # 返回分类输出和注意力权重
        if self.output_attention:
            return output, attns
        else:
            return output, l1_reg


class Config:
    """配置类，用于存储模型参数"""
    def __init__(self, seq_len, d_model=512, dropout=0.1, output_attention=True,
                use_norm=True, geomattn_dropout=0.5, alpha=0.5, kernel_size=4,
                embed='fixed', freq='h', factor=5, requires_grad=False, wv='db2',
                m=2, dec_in=6, e_layers=1, d_ff=2048, activation='gelu',
                num_classes=2):
        # 序列参数
        self.seq_len = seq_len  # 输入序列长度
        
        # 模型参数
        self.d_model = d_model  # 模型维度
        self.dropout = dropout  # dropout率
        self.output_attention = output_attention  # 是否输出注意力权重
        self.use_norm = use_norm  # 是否使用归一化
        self.geomattn_dropout = geomattn_dropout  # 几何注意力dropout率
        self.alpha = alpha  # 几何注意力中点积和楔形积的平衡系数
        self.kernel_size = kernel_size  # 小波变换的核大小
        self.embed = embed  # 嵌入类型
        self.freq = freq  # 频率类型
        self.factor = factor  # 注意力机制中的缩放因子
        self.requires_grad = requires_grad  # 是否需要梯度
        self.wv = wv  # 小波类型
        self.m = m  # 小波分解级别
        self.dec_in = dec_in  # 输入特征维度
        self.e_layers = e_layers  # 编码器层数
        self.d_ff = d_ff  # 前馈网络维度
        self.activation = activation  # 激活函数
        self.num_classes = num_classes  # 类别数 