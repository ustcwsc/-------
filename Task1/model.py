import torch
import torch.nn.functional as F

class CNN:
    def __init__(self, device):
        self.device = device
        
        # Kaiming 初始化
        def init_w(out_c, in_c, k):
            return torch.randn(out_c, in_c, k, k, device=device) * (2. / (in_c * k * k))**0.5

        # === 权重定义 (保持不变) ===
        self.conv1_weight = init_w(16, 1, 3)
        self.conv1_bias = torch.zeros(16, device=device)
        self.c2_w = init_w(16, 16, 3)
        self.c2_b = torch.zeros(16, device=device)

        self.conv2_weight = init_w(32, 16, 3)
        self.conv2_bias = torch.zeros(32, device=device)
        self.c4_w = init_w(32, 32, 3)
        self.c4_b = torch.zeros(32, device=device)

        self.c5_w = init_w(64, 32, 3)
        self.c5_b = torch.zeros(64, device=device)
        self.c6_w = init_w(64, 64, 3)
        self.c6_b = torch.zeros(64, device=device)

        # === 修改点在这里 ===
        # 原来是 64*28*28，现在输入320，经过3次2x2池化变成40
        self.flat_dim = 64 * 40 * 40 
        # ====================
        
        self.fc_weight = torch.randn(self.flat_dim, 1, device=device) * 0.01
        self.fc_bias = torch.zeros(1, device=device)

        # === 动量初始化 (保持不变) ===
        self.v_conv1_w = torch.zeros_like(self.conv1_weight)
        self.v_conv1_b = torch.zeros_like(self.conv1_bias)
        self.v_c2_w = torch.zeros_like(self.c2_w)
        self.v_c2_b = torch.zeros_like(self.c2_b)
        self.v_conv2_w = torch.zeros_like(self.conv2_weight)
        self.v_conv2_b = torch.zeros_like(self.conv2_bias)
        self.v_c4_w = torch.zeros_like(self.c4_w)
        self.v_c4_b = torch.zeros_like(self.c4_b)
        self.v_c5_w = torch.zeros_like(self.c5_w)
        self.v_c5_b = torch.zeros_like(self.c5_b)
        self.v_c6_w = torch.zeros_like(self.c6_w)
        self.v_c6_b = torch.zeros_like(self.c6_b)
        self.v_fc_w = torch.zeros_like(self.fc_weight)
        self.v_fc_b = torch.zeros_like(self.fc_bias)

    # ... forward 和 backward 方法保持不变 ...
    # (确保你使用的是包含 Momentum 的 backward 版本)
    def forward(self, x):
        # ... (代码不变)
        # 只需要确保这里的逻辑是通用的，
        # 只要 self.flat_dim 计算正确，.view(x.shape[0], -1) 会自动处理新的尺寸
        self.x = x
        # Block 1
        self.z1 = F.conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
        self.a1 = F.relu(self.z1)
        self.z2 = F.conv2d(self.a1, self.c2_w, self.c2_b, padding=1)
        self.a2 = F.relu(self.z2)
        self.p1, self.idx1 = F.max_pool2d(self.a2, 2, return_indices=True)

        # Block 2
        self.z3 = F.conv2d(self.p1, self.conv2_weight, self.conv2_bias, padding=1)
        self.a3 = F.relu(self.z3)
        self.z4 = F.conv2d(self.a3, self.c4_w, self.c4_b, padding=1)
        self.a4 = F.relu(self.z4)
        self.p2, self.idx2 = F.max_pool2d(self.a4, 2, return_indices=True)

        # Block 3
        self.z5 = F.conv2d(self.p2, self.c5_w, self.c5_b, padding=1)
        self.a5 = F.relu(self.z5)
        self.z6 = F.conv2d(self.a5, self.c6_w, self.c6_b, padding=1)
        self.a6 = F.relu(self.z6)
        self.p3, self.idx3 = F.max_pool2d(self.a6, 2, return_indices=True)

        self.flat = self.p3.view(x.shape[0], -1)
        self.z_fc = self.flat.matmul(self.fc_weight) + self.fc_bias
        self.out = torch.sigmoid(self.z_fc).squeeze(1)
        return self.out
    
    # ... backward 代码不变 ...
    def backward(self, y, lr, momentum=0.9):
        # ... (代码不变，只要前面的修改正确即可)
        B = y.shape[0]
        y = y.view(-1, 1)
        out = self.out.view(-1, 1)

        # === 梯度计算 (与之前相同) ===
        # 降低 pos_weight 到 2.0，减少梯度爆炸风险
        pos_weight = 2.0
        weights = torch.where(y == 1, torch.tensor(pos_weight, device=self.device), torch.tensor(1.0, device=self.device))
        
        # d_loss / d_z_fc
        dz_fc = (out - y) * weights

        # FC gradients
        dw_fc = self.flat.t().matmul(dz_fc) / B
        db_fc = dz_fc.mean(dim=0)
        d_p3 = dz_fc.matmul(self.fc_weight.t()).view(self.p3.shape)

        # Block 3 gradients
        d_a6 = F.max_unpool2d(d_p3, self.idx3, 2, output_size=self.a6.shape)
        d_z6 = d_a6 * (self.z6 > 0).float()
        dw6 = F.conv2d(self.a5.transpose(0,1), d_z6.transpose(0,1), padding=1).transpose(0,1) / B
        db6 = d_z6.sum(dim=(0,2,3)) / B
        d_a5 = F.conv_transpose2d(d_z6, self.c6_w, padding=1)

        d_z5 = d_a5 * (self.z5 > 0).float()
        dw5 = F.conv2d(self.p2.transpose(0,1), d_z5.transpose(0,1), padding=1).transpose(0,1) / B
        db5 = d_z5.sum(dim=(0,2,3)) / B
        d_p2 = F.conv_transpose2d(d_z5, self.c5_w, padding=1)

        # Block 2 gradients
        d_a4 = F.max_unpool2d(d_p2, self.idx2, 2, output_size=self.a4.shape)
        d_z4 = d_a4 * (self.z4 > 0).float()
        dw4 = F.conv2d(self.a3.transpose(0,1), d_z4.transpose(0,1), padding=1).transpose(0,1) / B
        db4 = d_z4.sum(dim=(0,2,3)) / B
        d_a3 = F.conv_transpose2d(d_z4, self.c4_w, padding=1)

        d_z3 = d_a3 * (self.z3 > 0).float()
        dw3 = F.conv2d(self.p1.transpose(0,1), d_z3.transpose(0,1), padding=1).transpose(0,1) / B
        db3 = d_z3.sum(dim=(0,2,3)) / B
        d_p1 = F.conv_transpose2d(d_z3, self.conv2_weight, padding=1)

        # Block 1 gradients
        d_a2 = F.max_unpool2d(d_p1, self.idx1, 2, output_size=self.a2.shape)
        d_z2 = d_a2 * (self.z2 > 0).float()
        dw2 = F.conv2d(self.a1.transpose(0,1), d_z2.transpose(0,1), padding=1).transpose(0,1) / B
        db2 = d_z2.sum(dim=(0,2,3)) / B
        d_a1 = F.conv_transpose2d(d_z2, self.c2_w, padding=1)

        d_z1 = d_a1 * (self.z1 > 0).float()
        dw1 = F.conv2d(self.x.transpose(0,1), d_z1.transpose(0,1), padding=1).transpose(0,1) / B
        db1 = d_z1.sum(dim=(0,2,3)) / B

        # === 动量更新 (Momentum Update) ===
        # 公式: v = m * v - lr * grad
        #       w = w + v
        
        def update(w, b, dw, db, vw, vb):
            vw.mul_(momentum).sub_(lr * dw)
            vb.mul_(momentum).sub_(lr * db)
            w.add_(vw)
            b.add_(vb)

        update(self.fc_weight, self.fc_bias, dw_fc, db_fc, self.v_fc_w, self.v_fc_b)
        update(self.c6_w, self.c6_b, dw6, db6, self.v_c6_w, self.v_c6_b)
        update(self.c5_w, self.c5_b, dw5, db5, self.v_c5_w, self.v_c5_b)
        update(self.c4_w, self.c4_b, dw4, db4, self.v_c4_w, self.v_c4_b)
        update(self.conv2_weight, self.conv2_bias, dw3, db3, self.v_conv2_w, self.v_conv2_b)
        update(self.c2_w, self.c2_b, dw2, db2, self.v_c2_w, self.v_c2_b)
        update(self.conv1_weight, self.conv1_bias, dw1, db1, self.v_conv1_w, self.v_conv1_b)