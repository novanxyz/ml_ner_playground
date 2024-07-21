"""
Tagging example.
"""
import argparse
import os
from pprint import pprint

from anago.tagger import Tagger
from anago.models import BiLSTMCRF, load_model
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from lang_util.spell import SpellCorrector


spell = SpellCorrector()



def main(args):
    print('Loading objects...')
    model = load_model(args.weights_file, args.params_file)
    it = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=it)

    print('Tagging a sentence...:' + args.sent)    
    test = spell.validate(args.sent)
    res = tagger.analyze(test)
    pprint(res)


if __name__ == '__main__':
    SAVE_DIR = os.path.join(os.path.dirname(__file__), '../models')
    parser = argparse.ArgumentParser(description='Tagging a sentence.')
    parser.add_argument('--sent', default='dok apakah khamilan ank prtama menyebabkan mual2 terus')
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocess.bin'))
    args = parser.parse_args()
    if args.model_dir :#
        model_dir =  os.path.abspath(os.path.join(SAVE_DIR,args.model_dir))
        args.weights_file   = os.path.join(model_dir,'weights.h5')
        args.params_file    = os.path.join(model_dir,'params.json')
        args.preprocessor_file = os.path.join(model_dir,'preprocessor.bin')
        
    main(args)
