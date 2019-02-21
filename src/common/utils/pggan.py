import math
import chainer.functions as F

def downsize_real(x_real, stage, max_stage=17):
    assert x_real.shape[2] == x_real.shape[3]
    input_size = x_real.shape[2]

    stage = min(stage, max_stage - 1e-8)
    alpha = stage - math.floor(stage)
    stage = math.floor(stage)

    if stage % 2 == 0:
        k = (stage - 2) // 2
        image_size = 4 * (2 ** (k + 1))

        assert image_size <= input_size
        scale = input_size // image_size

        result = x_real
        if scale > 1:
            result = F.average_pooling_2d(result, scale, scale, 0)

    else:
        k = (stage - 1) // 2
        image_size_low = 4 * (2 ** (k))
        image_size_high = 4 * (2 ** (k + 1))

        assert image_size_high <= input_size
        scale_low = input_size // image_size_low
        scale_high = input_size // image_size_high

        result_low = x_real
        result_high = x_real

        if scale_low > 1:
            result_low = F.unpooling_2d(
                F.average_pooling_2d(result_low, scale_low, scale_low, 0),
                2,
                2,
                0,
                outsize=(image_size_high, image_size_high))

        if scale_high > 1:
            result_high = F.average_pooling_2d(result_high, scale_high, scale_high, 0)

        result = (1 - alpha) * result_low + alpha * result_high

    return result