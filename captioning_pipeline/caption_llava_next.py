import torch
from PIL import Image
#from lavis.models import load_model_and_preprocess
import glob
import pickle as pkl
from tqdm import tqdm
import os
import argparse
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import nltk

# Ensure the punkt tokenizer is downloaded


nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type = str, default='./example_material')
    #parser.add_argument("--model_type", type = str, default='pretrain_flant5xxl', choices=['pretrain_flant5xxl', 'pretrain_flant5xl'])
    parser.add_argument("--use_qa", action="store_true")
    return parser.parse_args()
def remove_prompt(text):
    # Function to remove the prompt from the generated text
    return text.split('[/INST]')[-1].strip()  # Adjust based on the exact prompt format

def trim_to_complete_sentence(text):
    # Function to trim text to the last complete sentence
    sentences = nltk.tokenize.sent_tokenize(text)
    if sentences:
        complete_text = " ".join(sentences[:-1]) if not sentences[-1].endswith(('.', '!', '?')) else text
        return complete_text.strip()
    return text.strip()

def main(view_number):
    args = parse_args()

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    all_output = {}

    #name = 'blip2_t5'
    #model_type = args.model_type

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True) 
    model.to("cuda:0")
    processor = LlavaNextProcessor.from_pretrained(model_id)

    outfilename = f'{args.parent_dir}/Cap3D_captions/Cap3D_captions_view{view_number}.pkl'
    infolder = f'{args.parent_dir}/Cap3D_imgs/Cap3D_imgs_view{view_number}/*.png'
    
    if os.path.exists(outfilename):
        with open(outfilename, 'rb') as f:
            all_output = pkl.load(f)

    print("number of annotations so far",len(all_output))

    #model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)
    ct = 0

    all_files = glob.glob(infolder)
    all_imgs = [x for x in all_files if ".png" in x.split("_")[-1]]
    print("Total .png files found:", len(all_imgs))

    all_imgs = [x for x in all_imgs if x not in all_output]
    print("New images to process:", len(all_imgs))

    for filename in tqdm(all_imgs):
        # skip the images we have already generated captions
        if os.path.exists(outfilename):
                if os.path.basename(filename).split('.')[0] in all_output.keys():
                    continue
        try:
            #raw_image = Image.open(filename).convert("RGB")
            print(f"path = {filename}")
            image = Image.open(filename)
        except:
            print("file not work skipping", filename)
            continue

        #image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        if args.use_qa:
            # prompt = "Question: what object is in this image? Answer:"
            # object = model.generate({"image": image, "prompt": prompt})[0]
            # full_prompt = "Question: what is the structure and geometry of this %s?" % object
            # x = model.generate({"image": image, "prompt": full_prompt}, use_nucleus_sampling=True, num_captions=5)
            pass
        else:
            prompt = "[INST] <image>\nCan you describe the object strictly within 75 tokens? [/INST]"
            inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
            # autoregressively complete prompt to generate 5 captions

            output = model.generate(**inputs, max_new_tokens=75, do_sample=True, top_k=50, num_return_sequences=1)


            #x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=5)
        
        # Decode generated captions and remove the prompt
        captions = [remove_prompt(processor.decode(out, skip_special_tokens=True)) for out in output]

        # Ensure the captions end with complete sentences
        complete_captions = [trim_to_complete_sentence(caption) for caption in captions]

        all_output[os.path.basename(filename).split('.')[0]] = complete_captions
        print(f"Generated captions for image {filename}: {complete_captions}")

        #all_output[os.path.basename(filename).split('.')[0]] = [z for z in x]
        
        if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
            print(f"Saving output at count {ct} for image {filename}")
            with open(outfilename, 'wb') as f:
                pkl.dump(all_output, f)
            
            # print(filename)
            # print([z for z in x])

            # with open(outfilename, 'wb') as f:
            #     pkl.dump(all_output, f)
            
        ct += 1
    print(f"Saving final output after processing {ct} images.")
    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)

if __name__ == "__main__":
    for i in range(8):
        main(view_number=i)
