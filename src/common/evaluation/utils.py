import chainer

def get_classifer(clsf_type, clsf_path):
    from common.models import Inception
    if clsf_type == 'inception':
        model = Inception()
        args = {
                'get_feature': True,
                'scaled': True,
                'resize':True,
            }
    else:
        raise NotImplementedError
    model_path = clsf_path
    chainer.serializers.load_npz(model_path, model)
    return model, args