import torch
import torch.nn as nn
import torch.nn.functional as F
from .sciplex_model import sciplex3_model
from .gears_model import gears_model
from torchmetrics import TweedieDevianceScore


class TweedieLoss:
    def __init__(self, power=0):
        self.deviance_score = TweedieDevianceScore(power=power).cuda()

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return self.deviance_score(y_pred, y_true)


def orthogonal(input1: torch.Tensor, input2: torch.Tensor):
    # input1_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    # temp1 = input1.div(input1_norm.expand_as(input1) + 1e-6)

    # input2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    # temp2 = input2.div(input2_norm.expand_as(input2) + 1e-6)

    ortho_loss = torch.mean(
        torch.square(torch.diagonal(torch.matmul(input1, input2.t())))
    )

    return ortho_loss


class XTransferCDR(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()

        self.loss_autoencoder = nn.MSELoss()
        self.lambda_reco_1 = config["lambda_reco_1"]
        self.lambda_reco_2 = config["lambda_reco_2"]
        self.lambda_cross_1 = config["lambda_cross_1"]
        self.lambda_cross_2 = config["lambda_cross_2"]
        self.lambda_cont_1 = config["lambda_cont_1"]
        self.lambda_cont_2 = config["lambda_cont_2"]
        self.lambda_dis = config["lambda_dis"]
        self.lambda_orthogonal = config["lambda_orthogonal"]
        self.num_epoch = config["num_epoch"]
        self.encode_noise_factor = config["encode_noise_factor"]
        self.decode_noise_factor = config["decode_noise_factor"]
        self.mode = config["mode"]

        model_dict = {
            "sciplex3": sciplex3_model,
            "gears": gears_model,
        }

        self.encoderG_P, self.encoderG_S, self.decoderG = model_dict[config["dataset"]](
            config, dtype
        )

    def decoder(self, gene, pert) -> torch.Tensor:
        return self.decoderG(gene + pert)

    def add_noise(self, inputs: torch.Tensor, noise_factor=0.1):
        if (
            self.training
        ):
            noisy = inputs + torch.randn_like(inputs).cuda() * noise_factor
            # noisy = torch.clip(noisy, 0., 1.)
            return noisy
        return inputs

    def forward(
        self,
        controls_left: torch.Tensor,
        controls_right: torch.Tensor,
        treats_left: torch.Tensor,
        treats_right: torch.Tensor,
        # cross_left: torch.Tensor,
        # cross_right: torch.Tensor,
    ):
        treats_left_noise = self.add_noise(treats_left, self.encode_noise_factor)
        treats_right_noise = self.add_noise(treats_right, self.encode_noise_factor)

        pert_left = self.encoderG_P(treats_left_noise)
        pert_left_noise = self.add_noise(pert_left, self.decode_noise_factor)
        pert_right = self.encoderG_P(treats_right_noise)
        pert_right_noise = self.add_noise(pert_right, self.decode_noise_factor)

        base_left = self.encoderG_S(treats_left_noise)
        base_left_noise = self.add_noise(base_left, self.decode_noise_factor)
        base_right = self.encoderG_S(treats_right_noise)
        base_right_noise = self.add_noise(base_right, self.decode_noise_factor)

        if self.mode == "cell": 
            dis = torch.nn.functional.kl_div(
                F.log_softmax(base_left, dim=1),
                F.softmax(base_right, dim=1),
                reduction="batchmean",
            )
        elif self.mode == "drug":
            dis = torch.nn.functional.kl_div(
                F.log_softmax(pert_left, dim=1),
                F.softmax(pert_right, dim=1),
                reduction="batchmean",
            )
        else:  
            dis = 0

        orthogonal_loss = orthogonal(base_left, pert_left) + orthogonal(
            base_right, pert_right
        )

        base_left_G = self.decoderG(base_left_noise)
        base_right_G = self.decoderG(base_right_noise)
        loss_base_left = self.loss_autoencoder(base_left_G, controls_left)
        loss_base_right = self.loss_autoencoder(base_right_G, controls_right)

        if self.mode == "cell":
            treats_1 = self.decoder(base_left_noise, pert_left_noise)
            treats_2 = self.decoder(base_right_noise, pert_left_noise)
            treats_3 = self.decoder(base_left_noise, pert_right_noise)
            treats_4 = self.decoder(base_right_noise, pert_right_noise)
            loss_2 = self.loss_autoencoder(treats_left, treats_2)
            loss_3 = self.loss_autoencoder(treats_right, treats_3)

        elif self.mode == "drug":
            treats_1 = self.decoder(base_left_noise, pert_left_noise)
            treats_2 = self.decoder(base_left_noise, pert_right_noise)
            treats_3 = self.decoder(base_right_noise, pert_left_noise)
            treats_4 = self.decoder(base_right_noise, pert_right_noise)
            loss_2 = self.loss_autoencoder(treats_left, treats_2)
            loss_3 = self.loss_autoencoder(treats_right, treats_3)

        else:
            treats_1 = self.decoder(base_left_noise, pert_left_noise)
            treats_2 = self.decoder(base_right_noise, pert_left_noise)
            treats_3 = self.decoder(base_left_noise, pert_right_noise)
            treats_4 = self.decoder(base_right_noise, pert_right_noise)
            
        loss_1 = self.loss_autoencoder(treats_left, treats_1)
        loss_4 = self.loss_autoencoder(treats_right, treats_4)

        loss = loss_1 * self.lambda_reco_1 + loss_4 * self.lambda_reco_2

        loss += loss_2 * self.lambda_cross_1 + loss_3 * self.lambda_cross_2

        loss += loss_base_left * self.lambda_cont_1 + loss_base_right * self.lambda_cont_2

        loss += dis * self.lambda_dis + orthogonal_loss * self.lambda_orthogonal

        embedding = (
            pert_left,
            pert_right,
            base_left,
            base_right,
            treats_2.cpu(),
            treats_3.cpu(),
        )

        return loss, loss_1, loss_2, loss_3, loss_4, loss_base_left, loss_base_right, dis, orthogonal_loss, embedding

    def predict(
        self,
        treats_cell_a_with_pert_1,
        control_cell_b
    ):
        pert_1_for_cell_a = self.encoderG_P(treats_cell_a_with_pert_1)

        base_control_cell_b = self.encoderG_S(control_cell_b)

        treats_cell_b_with_pert_1_cross = self.decoder(base_control_cell_b, pert_1_for_cell_a)

        embedding = (
            # base_a_for_pert_1,
            pert_1_for_cell_a,
            base_control_cell_b
        )

        return (
            treats_cell_b_with_pert_1_cross,
            embedding,
        )