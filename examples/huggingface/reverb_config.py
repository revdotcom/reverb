# Following https://huggingface.co/docs/transformers/en/custom_models
import math
from typing import Dict, List, Optional
from transformers import PretrainedConfig
import numpy as np


def cmvn(means: List[float], variance: List[float], count: int):
    """ Calculate cmvn from stats

    Returns:
        a numpy array of [means, vars]
    """
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    # cmvn = np.array([means, variance])
    return [means, variance]


class ReverbConfig(PretrainedConfig):
    # Not sure what to put but also not reqruied: model_type = "encoderdecoder"
    model_type = "reverb_asr"
    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 10001,
        cmvn_mean_stat : List[float] = [33596438528.0, 35418329088.0, 39182106624.0, 41983324160.0, 44419112960.0, 46015381504.0, 46934564864.0, 47058870272.0, 47288012800.0, 47522979840.0, 48491438080.0, 49308729344.0, 50230493184.0, 50796900352.0, 51020386304.0, 51297456128.0, 51333586944.0, 51126181888.0, 51455569920.0, 50636410880.0, 49947033600.0, 50365546496.0, 49383075840.0, 49540546560.0, 49066065920.0, 49236889600.0, 48820707328.0, 49071112192.0, 48968024064.0, 49024458752.0, 49202397184.0, 49374433280.0, 49620660224.0, 49947111424.0, 50326310912.0, 50717818880.0, 51046891520.0, 51345678336.0, 51655733248.0, 51505459200.0, 51813666816.0, 51577262080.0, 51776524288.0, 51754237952.0, 51918598144.0, 52158758912.0, 52405276672.0, 52596776960.0, 52639731712.0, 52631220224.0, 52443103232.0, 52315619328.0, 52219695104.0, 52178399232.0, 52083040256.0, 52064792576.0, 51980918784.0, 51824164864.0, 51550973952.0, 51002216448.0, 50422747136.0, 49847754752.0, 49474338816.0, 48997863424.0, 48617009152.0, 48309174272.0, 48084140032.0, 48095608832.0, 47965765632.0, 47909335040.0, 47780065280.0, 47762370560.0, 47757099008.0, 47731314688.0, 47574110208.0, 47336361984.0, 47009054720.0, 46283513856.0, 44821860352.0, 42771775488.0],
        cmvn_var_stat: List[float] = [360475131904.0, 401487724544.0, 484368646144.0, 548414357504.0, 608912080896.0, 651613241344.0, 678013698048.0, 683624693760.0, 689524047872.0, 695375822848.0, 722376851456.0, 746773872640.0, 774244204544.0, 791678353408.0, 798920015872.0, 807307444224.0, 808713453568.0, 802957754368.0, 812319899648.0, 788076953600.0, 767619497984.0, 777970712576.0, 748566544384.0, 751065628672.0, 736340869120.0, 739872473088.0, 727466704896.0, 734006083584.0, 731017904128.0, 732582576128.0, 737590444032.0, 742469861376.0, 749455671296.0, 758746972160.0, 769666121728.0, 781107331072.0, 790730506240.0, 799342002176.0, 808164917248.0, 803454713856.0, 812040585216.0, 804632395776.0, 809866821632.0, 808861499392.0, 813548044288.0, 820701954048.0, 828343779328.0, 834335604736.0, 835754590208.0, 835251011584.0, 829192929280.0, 824705744896.0, 821224734720.0, 819399753728.0, 816182853632.0, 815243788288.0, 812578177024.0, 807846281216.0, 799796035584.0, 784661544960.0, 770915631104.0, 756696285184.0, 746462183424.0, 734193254400.0, 724980072448.0, 717529612288.0, 711156563968.0, 710358204416.0, 706386919424.0, 704228884480.0, 700537110528.0, 699519008768.0, 699025129472.0, 698035535872.0, 693109391360.0, 686047887360.0, 676213948416.0, 655917645824.0, 616676458496.0, 563932168192.0],
        cmvn_frame_num: int = 3519342927,
        encoder: str = "conformer",
        encoder_activation_type: str = "swish",
        encoder_attention_dropout_rate: float = 0.1,
        encoder_attention_heads: int = 8,
        encoder_causal: bool = True,
        encoder_cnn_module_kernel: int = 31,
        encoder_cnn_module_norm: str = "layer_norm",
        encoder_dropout_rate: float = 0.1,
        encoder_input_layer: str = "conv2d",
        encoder_linear_units: int = 2048,
        encoder_normalize_before: bool = True,
        encoder_num_blocks: int = 18,
        encoder_num_langs: int = 2,
        encoder_output_size: int = 640,
        encoder_pos_enc_layer_type: str = "rel_pos",
        encoder_positional_dropout_rate: float = 0.1,
        encoder_selfattention_layer_type: str = "rel_selfattn",
        encoder_use_cnn_module: bool = True,
        encoder_use_dynamic_chunk: bool = True,
        decoder: str = "lslbitransformer",
        decoder_attention_heads: int = 8,
        decoder_dropout_rate: float = 0.1,
        decoder_linear_units: int = 2048,
        decoder_num_blocks: int = 6,
        decoder_num_langs: int = 2,
        decoder_positional_dropout_rate: float = 0.1,
        decoder_r_num_blocks: int = 6,
        decoder_self_attention_dropout_rate: float = 0.1,
        decoder_src_attention_dropout_rate: float = 0.1,
        ctc_blank_id: int = 0,
        ctc_weight: float = 0.3,
        lsm_weight: float = 0.1,
        reverse_weight: float = 0.3,
        special_tokens: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = encoder
        self.encoder_activation_type = encoder_activation_type
        self.encoder_attention_dropout_rate = encoder_attention_dropout_rate
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_causal = encoder_causal
        self.encoder_cnn_module_kernel = encoder_cnn_module_kernel
        self.encoder_cnn_module_norm = encoder_cnn_module_norm
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_input_layer = encoder_input_layer
        self.encoder_linear_units = encoder_linear_units
        self.encoder_normalize_before = encoder_normalize_before
        self.encoder_num_blocks = encoder_num_blocks
        self.encoder_num_langs = encoder_num_langs
        self.encoder_output_size = encoder_output_size
        self.encoder_pos_enc_layer_type = encoder_pos_enc_layer_type
        self.encoder_positional_dropout_rate = encoder_positional_dropout_rate
        self.encoder_selfattention_layer_type = encoder_selfattention_layer_type
        self.encoder_use_cnn_module = encoder_use_cnn_module
        self.encoder_use_dynamic_chunk = encoder_use_dynamic_chunk
        self.decoder = decoder
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_linear_units = decoder_linear_units
        self.decoder_num_blocks = decoder_num_blocks
        self.decoder_num_langs = decoder_num_langs
        self.decoder_positional_dropout_rate = decoder_positional_dropout_rate
        self.decoder_r_num_blocks = decoder_r_num_blocks
        self.decoder_self_attention_dropout_rate = decoder_self_attention_dropout_rate
        self.decoder_src_attention_dropout_rate = decoder_src_attention_dropout_rate
        self.ctc_blank_id = ctc_blank_id
        self.ctc_weight = ctc_weight
        self.lsm_weight = lsm_weight
        self.reverse_weight = reverse_weight
        if special_tokens is None:
            special_tokens = {
                "<blank>": 0,
                "<sos>": 2,
                "<eos>": 2,
                "<unk>": 1,
            }
        self.special_tokens = special_tokens
        self.cmvn_mean, self.cmvn_istd = cmvn(cmvn_mean_stat, cmvn_var_stat, cmvn_frame_num)
        self.inputs_to_logits_ratio = 1
        super().__init__(**kwargs)
