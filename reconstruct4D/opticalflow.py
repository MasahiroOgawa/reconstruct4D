# %%
import torch
import os
import sys
sys.path.append('../unimatch')
import main_flow
import unimatch


class UnimatchFlow():
    '''
    compute optical flow using unimatch algorithm
    '''
    def __init__(self) -> None:
        flow_dir = '../unimatch/output/todaiura'
        flow_files = sorted([file for file in os.listdir(flow_dir) if file.endswith('.flo')])
        print(f"flow_files={flow_files}")

        # # treat args as a global variable
        # parser = main_flow.get_args_parser()
        # args = parser.parse_args()
        # print(args)

        # # load model
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = unimatch.UniMatch(feature_channels=args.feature_channels,
        #                     num_scales=args.num_scales,
        #                     upsample_factor=args.upsample_factor,
        #                     num_head=args.num_head,
        #                     ffn_dim_expansion=args.ffn_dim_expansion,
        #                     num_transformer_layers=args.num_transformer_layers,
        #                     reg_refine=args.reg_refine,
        #                     task=args.task).to(device)
        # print(model)

    def compute(self, img1, img2):
        '''
        compute optical flow from 2 consecutive images.
        '''
        pass


# %%