from argparse import ArgumentParser
import os
from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    filter_result = dict(Nutrient_deficiency = result['pred_class'])

    show_result_pyplot(model, args.img, filter_result)


def predict_api(config_file, checkpoint_file, input_dir, output_dir, device='cuda:0' ):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model( config_file, checkpoint_file, device=device )
    result = inference_model(model, input_dir)
    result = dict(Nutrient_deficiency = result['pred_class'])
    model.show_result( input_dir, result, out_file=os.path.join(output_dir, os.path.basename(input_dir)))
    return result


if __name__ == '__main__':
    main()
