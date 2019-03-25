# -*- coding: utf-8 -*-
# @Time    : 2019/3/14 18:40
# @Author  : uhauha2929
from pyrouge import Rouge155
import shutil
import os

ROUGE_PATH = '/home/yzhao/soft/RELEASE-1.5.5'


def clear_dir(dpath):
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)


def eval_rouge(results, golds):
    assert len(golds) == len(results)
    clear_dir('gold_summaries')
    clear_dir('result_summaries')
    r = Rouge155(ROUGE_PATH)
    r.system_dir = 'result_summaries'
    r.model_dir = 'gold_summaries'
    r.system_filename_pattern = 'result.(\\d+).txt'
    r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
    for i in range(len(golds)):
        output_gold = open('gold_summaries/gold.A.%d.txt' % i, 'w')
        output_result = open('result_summaries/result.%d.txt' % i, 'w')
        if isinstance(golds[i], list):
            output_gold.write('\n'.join(golds[i]))  # 文件中一行一个句子
        else:
            output_gold.write(golds[i])
        if isinstance(results[i], list):
            output_result.write('\n'.join(results[i]))
        else:
            output_result.write(results[i])

        output_gold.close()
        output_result.close()

    output = r.convert_and_evaluate(
        rouge_args='-e {}/data -n 2 -w 1.2 -a'.format(ROUGE_PATH))
    print(output)
    return r.output_to_dict(output)


if __name__ == '__main__':
    r = Rouge155(ROUGE_PATH)
    r.system_dir = '/home/yzhao/workspace/SentenceSummarization/result_summaries'
    r.model_dir = '/home/yzhao/workspace/SentenceSummarization/gold_summaries'
    r.system_filename_pattern = 'result.(\\d+).txt'
    r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
    output = r.convert_and_evaluate(rouge_args='-e {}/data -n 2 -w 1.2 -a'.format(ROUGE_PATH))
    print(output)
    output_dict = r.output_to_dict(output)
