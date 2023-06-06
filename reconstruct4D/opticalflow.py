# %%
import sys
sys.path.append('../unimatch')
import main_flow
import torch
import unimatch.unimatch as unimatch

'''
compute optical flow estimation.
'''
class UnimatchFlow():
    def __init__(self) -> None:
        # treat args as a global variable
        parser = main_flow.get_args_parser()
        args = parser.parse_args()
        print(args)

        # load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = unimatch.UniMatch(feature_channels=args.feature_channels,
                            num_scales=args.num_scales,
                            upsample_factor=args.upsample_factor,
                            num_head=args.num_head,
                            ffn_dim_expansion=args.ffn_dim_expansion,
                            num_transformer_layers=args.num_transformer_layers,
                            reg_refine=args.reg_refine,
                            task=args.task).to(device)
        print(model)
# %%