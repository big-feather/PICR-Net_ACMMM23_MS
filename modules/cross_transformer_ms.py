"""
@Project: PICR_Net
@File: modules/cross_transformer.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""

import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
# from torch import einsum
# from einops import rearrange,  repeat
import mindspore.ops as ops
from modules.swin_transformer_ms1 import Mlp
from utils.torch_operation import trunc_normal_
import math


def einsum_ms(att_z, z_v):
    matmul_op = ops.MatMul(transpose_b=True)  # 进行矩阵乘法，需要将第二个矩阵转置
    elementwise_op = ops.Mul()  # 元素相乘
    sum_op = ops.ReduceSum(keep_dims=False)  # 对指定维度求和

    # 执行操作
    matmul_result = matmul_op(att_z, z_v)
    elementwise_result = elementwise_op(matmul_result, att_z)
    z_out = sum_op(elementwise_result, axis=3)  # 在维度3上求和

    return z_out

def einsum_ms1(z_q, z_k):
    # 使用标准操作实现类似结果
    matmul_op = ops.BatchMatMul(transpose_b=True)  # 进行批量矩阵乘法，需要将第二个矩阵转置
    reshape_op = ops.Reshape()  # 重塑张量形状
    transpose_op = ops.Transpose()  # 转置张量
    expand_dims_op = ops.ExpandDims()  # 扩展维度

    # 执行操作
    matmul_result = matmul_op(z_q, z_k)
    reshaped_result = reshape_op(matmul_result, (-1, matmul_result.shape[-2], matmul_result.shape[-1]))
    transposed_result = transpose_op(reshaped_result, (0, 2, 1))
    expanded_result = expand_dims_op(transposed_result, 2)

    return expanded_result

class RelationModel_double(nn.Cell):
    def __init__(self, dim, heads, dim_head, dropout):
        super(RelationModel_double, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv_z = nn.Dense(dim, inner_dim * 3, has_bias=False)
        self.to_out_z = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

        self.to_qkv_z1 = nn.Dense(dim, inner_dim * 3, has_bias=False)
        self.to_out_z1 = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

    def construct(self, x, y):
        b, p, d = x.shape# [1, 784, 192] [1, 196, 384] [1, 49, 768] [1, 49, 768]
        h = self.heads

        # print(x.shape,y.shape)

        rgb_guidence = ops.avg_pool1d(x.permute(0, 2, 1), kernel_size=p)
        rgb_guidence = rgb_guidence.permute(0, 2, 1).broadcast_to((b, p, d))

        depth_guidence = ops.avg_pool1d(y.permute(0, 2, 1), kernel_size=p)
        depth_guidence = depth_guidence.permute(0, 2, 1).broadcast_to((b, p, d))

        z = mindspore.ops.stack([x, y, rgb_guidence, depth_guidence], axis=2)  # [b, p, 4, d]
        z_qkv = self.to_qkv_z(z).chunk(3, axis=-1)  # List: 3 * [b, p, 4, d']

        z_q, z_k, z_v = z_qkv
        z_q = z_q.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_k = z_k.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_v = z_v.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        #t.re

        # z_q, z_k, z_v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv)
        # dots_z = ops.einsum('b p h i d, b p h j d -> b p h i j', z_q, z_k) * self.scale
        matmul_op = ops.BatchMatMul(transpose_b=True)  # 进行批量矩阵乘法，需要将第二个矩阵转置
        dots_z = matmul_op(z_q, z_k) * self.scale
        # dots_z = einsum_ms1(z_q, z_k) * self.scale
        atten_mask = mindspore.Tensor([[0., 0., 0., -100.],
                                   [0., 0., -100., 0.],
                                   [0., -100., 0., 0.],
                                   [-100., 0., 0., 0.]])
        attn_z = F.softmax(dots_z + atten_mask, axis=-1)  # affinity matrix

        # z_out = ops.einsum('b p h i j, b p h j d -> b p h i d', attn_z, z_v)
        b, p, h, i, j = attn_z.shape
        elementwise_op = mindspore.ops.BatchMatMul()
        z_v = z_v.permute(0, 1, 2, 4, 3).reshape(-1, 4, d//h)
        attn_z = attn_z.permute(0, 1, 2, 4, 3).reshape(-1, 4, 4)
        z_out = elementwise_op(attn_z, z_v)
        # z_out = einsum_ms(attn_z, z_v)
        # z_out = rearrange(z_out, 'b p h n d -> b p n (h d)')
        z_out = z_out.reshape(b, p, h, i, d//h).permute(0, 1, 3, 2, 4).reshape(b, p, -1, d)
        z_out = self.to_out_z(z_out)

        z1 = z_out  # [b, p, 4, d]
        z_qkv1 = self.to_qkv_z1(z1).chunk(3, axis=-1)  # List: 3 * [b, p, 4, d']
        z_q1, z_k1, z_v1 = z_qkv1
        z_q1 = z_q1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_k1 = z_k1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_v1 = z_v1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)




        # z_q1, z_k1, z_v1 = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv1)
        # dots_z1 = ops.einsum('b p h i d, b p h j d -> b p h i j', z_q1, z_k1) * self.scale
        matmul_op = ops.BatchMatMul(transpose_b=True)  # 进行批量矩阵乘法，需要将第二个矩阵转置
        dots_z1 = matmul_op(z_q1, z_k1) * self.scale
        atten_mask1 = mindspore.Tensor([[0., -100., 0., -100.],
                                    [-100., 0., -100., 0.],
                                    [-100., -100., -100., -100.],
                                    [-100., -100.,-100., -100.]])
        attn_z1 = F.softmax(dots_z1 + atten_mask1, axis=-1)  # affinity matrix

        # z_out1 = ops.einsum('b p h i j, b p h j d -> b p h i d', attn_z1, z_v1)
        elementwise_op = mindspore.ops.BatchMatMul()
        z_v1 = z_v1.permute(0, 1, 2, 4, 3).reshape(-1, 4, d//h)
        attn_z1 = attn_z1.permute(0, 1, 2, 4, 3).reshape(-1, 4, 4)
        z_out1 = elementwise_op(attn_z1, z_v1)
        z_out1 = z_out1.reshape(b, p, h, i, d//h).permute(0, 1, 3, 2, 4).reshape(b, p, -1, d)
        z_out1 = self.to_out_z1(z_out1)
        x1, y1 = z_out1[:, :, :2, :].chunk(2, axis=2)
        x1, y1 = x1.squeeze(axis=-2), y1.squeeze(axis=-2)

        return x1, y1






class RelationModel_side_double(nn.Cell):
    def __init__(self, dim, heads, dim_head, dropout):
        super(RelationModel_side_double, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv_z = nn.Dense(dim, inner_dim * 3, has_bias=False)
        self.to_out_z = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

        self.to_qkv_z1 = nn.Dense(dim, inner_dim * 3, has_bias=False)
        self.to_out_z1 = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

    def construct(self, x, y, s1):
        b, p, d = x.shape
        h = self.heads
        # s = s.detach()
        # B, C, H, W = f.shape
        s = s1
        s[s > 0.5] = 1
        s[s <= 0.5] = 1e-8
        v_s_sum =ops.sum(s.reshape(b, p), dim=1, keepdim=True)
        v_s_sum =ops.BroadcastTo(shape=(b, d))(v_s_sum)
        # v_s_sum = repeat(, 'b () -> b d', d=d)
        # print('*******************')
        # for pp in range(p):
        #     print(v_s_sum[0,pp])
        # print('*******************')

        rgb_guidence = ops.sum(ops.mul(x, s.reshape(b, p,1)).reshape(b, p, d), dim=1) / (v_s_sum + 1e-8)
        # v_ns = torch.sum(torch.mul(f, m2).reshape(B, C, H * W), dim=2) / (v_ns_sum + 1e-8)
        # print(v_s_sum)
        # exit(1)
        # rgb_guidence = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=p)

        rgb_guidence = rgb_guidence.reshape(b, 1, d).broadcast_to((b, p, d))

        depth_guidence = ops.sum(ops.mul(y, s.reshape(b, p,1)).reshape(b, p, d), dim=1) / (v_s_sum + 1e-8)
        depth_guidence = depth_guidence.reshape(b, 1, d).broadcast_to((b, p, d))

        z = ops.stack([x, y, rgb_guidence, depth_guidence], axis=2)  # [b, p, 4, d]
        z_qkv = self.to_qkv_z(z).chunk(3, axis=-1)  # List: 3 * [b, p, 4, d']

        z_q, z_k, z_v = z_qkv
        z_q = z_q.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_k = z_k.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_v = z_v.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        # t.re

        # z_q, z_k, z_v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv)
        # dots_z = ops.einsum('b p h i d, b p h j d -> b p h i j', z_q, z_k) * self.scale
        matmul_op = ops.BatchMatMul(transpose_b=True)  # 进行批量矩阵乘法，需要将第二个矩阵转置
        dots_z = matmul_op(z_q, z_k)* self.scale
        # dots_z = einsum_ms1(z_q, z_k) * self.scale
        atten_mask = mindspore.Tensor([[0., 0., 0., -100.],
                                       [0., 0., -100., 0.],
                                       [0., -100., 0., 0.],
                                       [-100., 0., 0., 0.]])
        attn_z = F.softmax(dots_z + atten_mask, axis=-1)  # affinity matrix

        # z_out = ops.einsum('b p h i j, b p h j d -> b p h i d', attn_z, z_v)
        elementwise_op = mindspore.ops.BatchMatMul()
        b, p, h, i, j = attn_z.shape
        z_v = z_v.permute(0, 1, 2, 4, 3).reshape(-1, 4, d//h)
        attn_z = attn_z.permute(0, 1, 2, 4, 3).reshape(-1, 4, 4)
        z_out = elementwise_op(attn_z, z_v)
        # z_out = einsum_ms(attn_z, z_v)
        # z_out = rearrange(z_out, 'b p h n d -> b p n (h d)')
        z_out = z_out.reshape(b, p, h, i, d//h).permute(0, 1, 3, 2, 4).reshape(b, p, -1, d)
        z_out = self.to_out_z(z_out)
        x, y = z_out[:, :, :2, :].chunk(2, axis=2)
        x, y = x.squeeze(axis=-2), y.squeeze(axis=-2)

        z1 = z_out  # [b, p, 4, d]
        z_qkv1 = self.to_qkv_z1(z1).chunk(3, axis=-1)  # List: 3 * [b, p, 4, d']

        z_q1, z_k1, z_v1 = z_qkv1
        z_q1 = z_q1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_k1 = z_k1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)
        z_v1 = z_v1.view(b, p, -1, h, d//h).permute(0, 1, 3, 2, 4)

        # z_q1, z_k1, z_v1 = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=h), z_qkv1)
        # dots_z1 = ops.einsum('b p h i d, b p h j d -> b p h i j', z_q1, z_k1) * self.scale
        matmul_op = ops.BatchMatMul(transpose_b=True)  # 进行批量矩阵乘法，需要将第二个矩阵转置
        dots_z1 = matmul_op(z_q1, z_k1) * self.scale
        atten_mask1 = mindspore.Tensor([[0., -100., 0., -100.],
                                        [-100., 0., -100., 0.],
                                        [-100., -100., -100., -100.],
                                        [-100., -100., -100., -100.]])
        attn_z1 = F.softmax(dots_z1 + atten_mask1, axis=-1)  # affinity matrix
        elementwise_op = mindspore.ops.BatchMatMul()
        z_v1 = z_v1.permute(0, 1, 2, 4, 3).reshape(-1, 4, d//h)
        attn_z1 = attn_z1.permute(0, 1, 2, 4, 3).reshape(-1, 4, 4)
        z_out1 = elementwise_op(attn_z1, z_v1)
        # z_out = einsum_ms(attn_z, z_v)
        # z_out1 = ops.einsum('b p h i j, b p h j d -> b p h i d', attn_z1, z_v1)
        z_out1 = z_out1.reshape(b, p, h, i, d//h).permute(0, 1, 3, 2, 4).reshape(b, p, -1, d)
        z_out1 = self.to_out_z1(z_out1)
        x1, y1 = z_out1[:, :, :2, :].chunk(2, axis=2)
        x1, y1 = x1.squeeze(axis=-2), y1.squeeze(axis=-2)

        return x1, y1


class PreNorm(nn.Cell):
    def __init__(self, dim, fn, dual=False):
        super().__init__()
        self.dual = dual
        self.norm1 = nn.LayerNorm([dim])
        self.norm2 = nn.LayerNorm([dim]) if dual else None
        self.fn = fn

    def construct(self, x, *args, **kwargs):
        if self.dual:
            y = args[0]
            return self.fn(self.norm1(x), self.norm2(y), *args[1:], **kwargs)

        return self.fn(self.norm1(x), *args, **kwargs)



class PointFusion(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                PreNorm(dim, RelationModel_double(dim, heads, dim_head, dropout=dropout), dual=True),
                PreNorm(dim, Mlp(dim, dim)),
                PreNorm(dim, Mlp(dim, dim))
            ]))
        self.fusion = PreNorm(dim * 2, nn.Dense(dim * 2, dim))

    def construct(self, x, y):
        for rm, mlp1, mlp2 in self.layers:
            rm_x, rm_y = rm(x, y)
            rm_x, rm_y = x + rm_x, y + rm_y
            mlp_x = mlp1(rm_x) + rm_x
            mlp_y = mlp2(rm_y) + rm_y
        out = ops.cat([mlp_x, mlp_y], axis=-1)
        out = self.fusion(out)

        return out






class PointFusion_side(nn.Cell):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                PreNorm(dim, RelationModel_side_double(dim, heads, dim_head, dropout=dropout), dual=True),
                PreNorm(dim, Mlp(dim, dim)),
                PreNorm(dim, Mlp(dim, dim))
            ]))
        self.fusion = PreNorm(dim * 2, nn.Dense(dim * 2, dim))

    def construct(self, x, y, s):
        for i,(rm, mlp1, mlp2) in enumerate(self.layers):

            rm_x, rm_y = rm(x, y,s)
            rm_x, rm_y = x + rm_x, y + rm_y
            mlp_x = mlp1(rm_x) + rm_x
            mlp_y = mlp2(rm_y) + rm_y
        out = mindspore.ops.cat([mlp_x, mlp_y], axis=-1)
        out = self.fusion(out)

        return out


if __name__ == '__main__':
    pass
    # x = torch.randn(1, 3, 128)
    # y = torch.randn(1, 3, 128)
    # model = CrossTransformer(128, 2, 4, 64, 64, 128)
    # out = model(x, y)
