from copy import copy
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torchmetrics import Metric as LightningMetric
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from pytorch_forecasting.models.base_model import BaseModelWithCovariates
from pytorch_forecasting.models.nn import LSTM, MultiEmbedding


class TemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(
            self,
            hidden_size: int = 16,
            lstm_layers: int = 2,
            dropout: float = 0.1,
            output_size: Union[int, List[int]] = 7,
            loss: MultiHorizonMetric = None,
            attention_head_size: int = 4,
            max_encoder_length: int = 10,
            static_categoricals: List[str] = [],
            static_reals: List[str] = [],
            time_varying_categoricals_encoder: List[str] = [],
            time_varying_categoricals_decoder: List[str] = [],
            categorical_groups: Dict[str, List[str]] = {},
            time_varying_reals_encoder: List[str] = [],
            time_varying_reals_decoder: List[str] = [],
            x_reals: List[str] = [],
            x_categoricals: List[str] = [],
            hidden_continuous_size: int = 8,
            hidden_continuous_sizes: Dict[str, int] = {},
            embedding_sizes: Dict[str, Tuple[int, int]] = {},
            embedding_paddings: List[str] = [],
            embedding_labels: Dict[str, np.ndarray] = {},
            learning_rate: float = 1e-3,
            log_interval: Union[int, float] = -1,
            log_val_interval: Union[int, float] = None,
            log_gradient_flow: bool = False,
            reduce_on_plateau_patience: int = 1000,
            monotone_constaints: Dict[str, int] = {},
            share_single_variable_networks: bool = False,
            causal_attention: bool = True,
            logging_metrics: nn.ModuleList = None,
            **kwargs,
    ):
        """
                Args:

            hidden_size: hidden size of network which is its main hyperparameter and can range from 8 to 512
            lstm_layers: number of LSTM layers (2 is mostly optimal)
            dropout: dropout rate
            output_size: number of outputs (e.g. number of quantiles for QuantileLoss and one target or list
                of output sizes).
            loss: loss function taking prediction and targets
            attention_head_size: number of attention heads (4 is a good default)
            max_encoder_length: length to encode (can be far longer than the decoder length but does not have to be)
            static_categoricals: names of static categorical variables
            static_reals: names of static continuous variables
            time_varying_categoricals_encoder: names of categorical variables for encoder
            time_varying_categoricals_decoder: names of categorical variables for decoder
            time_varying_reals_encoder: names of continuous variables for encoder
            time_varying_reals_decoder: names of continuous variables for decoder
            categorical_groups: dictionary where values
                are list of categorical variables that are forming together a new categorical
                variable which is the key in the dictionary
            x_reals: order of continuous variables in tensor passed to forward function
            x_categoricals: order of categorical variables in tensor passed to forward function
            hidden_continuous_size: default for hidden size for processing continous variables (similar to categorical
                embedding size)
            hidden_continuous_sizes: dictionary mapping continuous input indices to sizes for variable selection
                (fallback to hidden_continuous_size if index is not in dictionary)
            embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
                embedding size
            embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
            embedding_labels: dictionary mapping (string) indices to list of categorical labels
            learning_rate: learning rate
            log_interval: log predictions every x batches, do not log if 0 or less, log interpretation if > 0. If < 1.0
                , will log multiple entries per batch. Defaults to -1.
            log_val_interval: frequency with which to log validation set metrics, defaults to log_interval
            log_gradient_flow: if to log gradient flow, this takes time and should be only done to diagnose training
                failures
            reduce_on_plateau_patience (int): patience after which learning rate is reduced by a factor of 10
            monotone_constaints (Dict[str, int]): dictionary of monotonicity constraints for continuous decoder
                variables mapping
                position (e.g. ``"0"`` for first position) to constraint (``-1`` for negative and ``+1`` for positive,
                larger numbers add more weight to the constraint vs. the loss but are usually not necessary).
                This constraint significantly slows down training. Defaults to {}.
            share_single_variable_networks (bool): if to share the single variable networks between the encoder and
                decoder. Defaults to False.
            causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
                predictions. Defaults to True.
            logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are logged during training.
                Defaults to nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
            **kwargs: additional arguments to :py:class:`~BaseModel`.
        """
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = QuantileLoss()
        self.save_hyperparameters()  # 这个函数的作用就是保存类的参数，后面可以使用self.hparams访问每一个具体的参数，可以使用点符号访问
        # 这里没有保存损失函数的，后面会将损失函数作为一个模块单独保存
        assert isinstance(loss, LightningMetric), "Loss has to be a PyTorch Lightning `Metric`"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,

        )



















    def forward(self, x: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[
        str, Union[torch.Tensor, List[torch.Tensor]]]:
        pass