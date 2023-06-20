import subprocess
import os
from tokenizers import (
    Regex,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer
)
from tokenizers.processors import TemplateProcessing
import params as pr


def get_file(folder, wiki_file, files):
    if ''.join([folder, wiki_file, '-v1.zip']) not in files:
        try:
            subprocess.run(['wget', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip'])
        except Exception as e:
            print(e)
        subprocess.run(['curl', 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip', '--output', f'{wiki_file}-v1.zip'])


    if ''.join([folder, wiki_file]) not in files:
        try:
            subprocess.run(['unzip', f'{wiki_file}-v1.zip'])
        except Exception as e:
            print(e)
        subprocess.run(['tar', '-xf', f'{wiki_file}-v1.zip'])




def main():
    wiki_file = 'wikitext-103-raw'
    cur_folder = pr.current_folder
    files = os.listdir(cur_folder)
    
    
    get_file(cur_folder, wiki_file, files)
    
    
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex(' {2,}'), ' ')
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    special_tokens = ['<cls>', '<sep>', '<unk>', '<pad>', '<mask>', '<s>', '</s>']
    trainer = trainers.UnigramTrainer(vocab_size=pr.vocab_size, special_tokens=special_tokens, unk_token='<unk>')

    files = [f'{wiki_file}/wiki.{split}.raw' for split in ['test', 'train', 'valid']]

    tokenizer.train(files, trainer)

    cls_token_id = tokenizer.token_to_id('<cls>')
    sep_token_id = tokenizer.token_to_id('<sep>')

    tokenizer.post_processor = TemplateProcessing(
        single='$A:0 <sep>:0 <cls>:2',
        pair='$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2',
        special_tokens=[('<sep>', sep_token_id), ('<cls>', cls_token_id)]
    )


    tokenizer.save(f'{pr.token_filename}.json')

