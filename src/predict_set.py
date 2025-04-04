import sys
import os
sys.path.append('/')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from dataloader import LazySupervisedDataset,data_args
import argparse
import torch
import json
from tqdm import tqdm

def save_lst_to_jsonl(lst, filename):
    """Saves a Pandas DataFrame to a JSONL file.

    Args:
        df: The DataFrame to save.
        filename: The name of the JSONL file to create.
    """
    with open(filename, 'w') as f:
        for ele in lst:
            json.dump(ele, f)  # Convert each row to a dictionary and write as JSON
            f.write('\n')

def predict_set_data(args):

    print('Load Model...\n')
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = args.model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print('Load Dataset...\n')
    dataset_test = LazySupervisedDataset(
        data_path = args.dataset_path,
        data_args = data_args
    )
    # len(dataset_test)
    # print(type(dataset_test))

    # Extract model name from path for the output filename
    model_name = args.model_id.split('/')[-1]

    if not os.path.exists(args.res_folder):
        os.makedirs(args.res_folder)
    count = 0
    # Use the model name in the output filename
    output_file = f'{args.res_folder}/{model_name}.jsonl'

    print('Start Predict...')
    with open(output_file, 'w', encoding='utf-8') as f:
        for data_point in tqdm(dataset_test):
            # Audio-visual Inference
            conversation =  data_point['conversation']
            label = data_point['label']
            video_path = data_point['video_path']

            inputs = processor(conversation=conversation, return_tensors="pt")
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            output_ids = model.generate(**inputs, max_new_tokens=128)
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            os.remove(video_path)
            count+=1
            row = {'label':label,'predict':response}
            # break
            f.write(json.dumps(row) + '\n')

    print(f"Results saved to: {output_file}")

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='jsonl test set path',default='/workspace/videollama3_smes/data/test.jsonl')
    parser.add_argument('--model-id', help='', required=False, default="DAMO-NLP-SG/VideoLLaMA3-2B")
    parser.add_argument('--res-folder', help='path to file .jsonl predict', default='./eval_test/result')

    args = parser.parse_args()

    print('num answer : ', predict_set_data(args))