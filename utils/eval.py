import torch
import numpy as np
from tqdm import tqdm

import openai

from sentence_transformers import SentenceTransformer, util


def evaluation(logger, model, val_loader, affordance):
    num_classes = len(affordance)
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(len(affordance))]
    total_correct_class = [0 for _ in range(len(affordance))]
    total_iou_deno_class = [0 for _ in range(len(affordance))]
    with torch.no_grad():
        model.eval()
        for i,  temp_data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):

            (data, _, label, _, _) = temp_data

            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            label = torch.squeeze(label).cpu().numpy()
            B = label.shape[0]
            N = label.shape[1]
            
            afford_pred = model(data, affordance)
            afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()
            afford_pred = np.argmax(afford_pred, axis=2)
        
            correct = np.sum((afford_pred == label))
            total_correct += correct
            total_seen += (B * N)
            for i in range(num_classes):
                total_seen_class[i] += np.sum((label == i))
                total_correct_class[i] += np.sum((afford_pred == i) & (label == i))
                total_iou_deno_class[i] += np.sum((afford_pred == i) | (label == i))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        logger.cprint('eval point avg class IoU: %f' % (mIoU))
        logger.cprint('eval point accuracy: %f' % (total_correct / float(total_seen)))
        logger.cprint('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(num_classes):
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    affordance[l], total_correct_class[l] / float(total_iou_deno_class[l]))
        logger.cprint(iou_per_class_str)
    return mIoU


def test_clpp(logger, model, val_loader, affordance, prompt):
    with torch.no_grad():
        model.eval()
        for i, temp_data in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
            (data, _, label, _, model_class) = temp_data

            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            label = torch.squeeze(label).cpu().numpy()
            B = label.shape[0]
            N = label.shape[1]
            
            afford_pred = model.get_logits(data, prompt)
            print(f'the shape of afford_pred: {afford_pred.shape}')
            print(f'afford_pred: {afford_pred}')
            afford_pred = afford_pred.squeeze(-1).cpu().numpy()
            afford_pred = np.argmax(afford_pred)
            print(f'the shape of afford_pred: {afford_pred.shape}')
            print(f'afford_pred: {afford_pred}')

            print(f'prompt: {prompt}')
            print(f'object class: {model_class}')
            print(f'Best object: {model_class[afford_pred]}')


def get_best_obj(logger, model, test_loader, affordance, prompt, batch_size=5):
    num_classes = len(affordance)
    total_seen_class = torch.zeros(batch_size, num_classes)
    with torch.no_grad():
        model.eval()
        for i,  temp_data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):

            (data, _, label, _, model_class) = temp_data

            # prompt = "It can contain some objects or water"

            data, label = data.float().cuda(), label.float().cuda()
            data = data.permute(0, 2, 1)
            label = torch.squeeze(label).cpu().numpy()
            B = label.shape[0]
            N = label.shape[1]
            
            afford_pred = model(data, affordance)
            afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()
            afford_pred = np.argmax(afford_pred, axis=2)
            
            for j in range(num_classes):
                total_seen_class[:, j] += np.sum((afford_pred == j), axis=1)
            
            top_k = 3
            top_k_idx = torch.topk(total_seen_class, top_k, dim=1).indices

            # print(f'top_k_idx: {top_k_idx}')

            target_affordances_idx = get_affordance_transformer(prompt, affordance, top_k)
            # target_affordances_idx = [affordance.index(aff) for aff in target_affordances]

            best_obj_idx = match_affordances(top_k_idx, target_affordances_idx)

            print(f'prompt: {prompt}')
            print(f'object class: {model_class}')
            print(f'Best object: {model_class[best_obj_idx]}')


def get_affordance_gpt(prompt, affordance, top_k):
    max_retry = 3

    def get_response(msg):
        for _ in range(max_retry):
            # try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                # model='gpt-4',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                    ],
                temperature=0.001,
                max_tokens=1000,
                api_key = "",  # Add your API key here
            )
            return response.choices[0]["message"]["content"]

    openai.api_base = "https://api.openai-sb.com/v1"

    ques = (
        "I have a description of an object and a list of potential affordances. "
        "The object description is: '" + prompt + "' "
        "The list of 18 affordances is: ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable', "
        "'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']. "
        "Based on the description, select the " + str(top_k) + " affordances that most likely apply to this object. You only need to provide the numbers of the affordances in the list above (1 ~ 18), and split them by space."
    )

    response = get_response(ques)
    indexes = [int(i) for i in response.split()]
    return indexes


def get_affordance_transformer(prompt, affordance, top_k):

    model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda')

    description_embedding = model.encode(prompt, convert_to_tensor=True).to('cuda')
    affordance_embeddings = model.encode(affordance, convert_to_tensor=True).to('cuda')

    similarities = util.cos_sim(description_embedding, affordance_embeddings)
    top_3_indices = similarities.argsort(descending=True)
    top_3_indices = top_3_indices.squeeze(0).cpu().numpy()[:top_k]
    # print(f'top_3_indices: {top_3_indices}, top_k: {top_k}')
    top_3_affordances = [affordance[i] for i in top_3_indices]

    print("Top 3 Affordances:", top_3_affordances)

    return top_3_indices


def match_affordances(top_k_idx, target_affordances_idx):
    from scipy.stats import kendalltau

    def kendall_distance(ground_truth, pred):
        _, tau = kendalltau(ground_truth, pred)
        return tau

    distances = [kendall_distance(target_affordances_idx, pred) for pred in top_k_idx]

    best_match_idx = distances.index(max(distances))

    return best_match_idx

            