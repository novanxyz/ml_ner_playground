"""
Example from training to saving.
"""
import argparse
import os
import time
from pprint import pprint

from anago.utils import load_data_and_labels
from anago.models import BiLSTMCRF,load_model,save_model
from anago.preprocessing import IndexTransformer#, ELMoTransformer
from anago.trainer import Trainer
from anago.tagger import Tagger
from lang_util.spell import SpellCorrector


spell = SpellCorrector()

def main(args):
    print('Loading dataset...')
    x_train, y_train = load_data_and_labels(args.train_data)
    x_valid, y_valid = load_data_and_labels(args.valid_data)
    
    # apply spell corrector befor training
    if args.correct_spell:
        print('apply spell corrector')
        start_time = time.time()
        x_train = [ spell.validate(" ".join(x)).split()  for x in x_train ]
        x_valid = [ spell.validate(" ".join(x)).split() for x in x_valid ]
        elapsed_time = time.time() - start_time
        print('apply spell corrector:' + str(elapsed_time))

    for yt in y_train:
        for idx, yx in enumerate(yt):
            if " " in yx:
                yx = yx.split(" ")
                yx.sort()
                yt[idx] = " ".join( yx )

    for yv in y_valid:
        for idx, yx in enumerate(yv):
            if " " in yx:
                yx = yx.split(" ")
                yx.sort()
                yv[idx] = " ".join( yx )

    Y = set()
    for yt in y_train:
        for y in  (list(set(yt))):
            Y.add( y )

    print( sorted(Y) )

    print('Transforming datasets...')
    p = IndexTransformer(use_char=args.no_char_feature)
    p.fit(x_train, y_train)

    print ("Label size:",str(p.label_size) , len(Y) )
    pprint(p._label_vocab._token2id)

    print('Building a model.')
    model = BiLSTMCRF(char_embedding_dim=args.char_emb_size,
                      word_embedding_dim=args.word_emb_size,
                      char_lstm_size=args.char_lstm_units,
                      word_lstm_size=args.word_lstm_units,
                      char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      dropout=args.dropout,
                      use_char=args.no_char_feature,
                      use_crf=args.no_use_crf)
    model, loss = model.build()
    model.compile(loss=loss, optimizer='adam')

    print('Training the model...')
    trainer = Trainer(model, preprocessor=p)
    trainer.train(x_train, y_train, x_valid, y_valid,epochs=args.epochs)

    print('Saving the model...')
    save_model(model,args.weights_file, args.params_file)    
    p.save(args.preprocessor_file)    
    
    #model.score(x_valid,y_valid)
    
    test = "kenapa bs hidrosefalusdok anak pertama sya dibilang dokter kena hidrosefalus ada cairan yang berkumpul di bagian otaknya ini sebenarnya penyebabnya apa ya dok"
    test = "bayi bahayakah dok jika tidak membuat bayi selsaai menyusui? soalnya kan dok bayi sy asal di tepuk belakangnya dia tersentak dan tak mau tidur lagi"
    test = "dok apakah kehamilan anak pertama menyebabkan mual2 terus"
    test = "dok saya mau tanya bahan alami untuk obatin penyakit herpes apa"
    test = "ibu mertua sya biduran udah seminggu, obatnya apa ya dok?"
    test = spell.validate(test)
    tagger =  Tagger(model,p)
    res = tagger.analyze(test)
    pprint(res)


if __name__ == '__main__':
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))
    parser = argparse.ArgumentParser(description='Training a model')

    parser.add_argument('--train_data', default=os.path.join(DATA_DIR, 'train.txt'), help='training data')
    parser.add_argument('--valid_data', default=os.path.join(DATA_DIR, 'valid.txt'), help='validation data')
    parser.add_argument('--save_dir', default=MODELS_DIR)

    parser.add_argument('--weights_file', default=None, help='weights file')
    parser.add_argument('--params_file', default=None, help='parameter file')
    parser.add_argument('--preprocessor_file',default=None, help='preprocessor_file')
    # Training parameters
    parser.add_argument('--loss', default='categorical_crossentropy', help='loss')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--checkpoint_path', default=MODELS_DIR, help='checkpoint path')
    parser.add_argument('--log_dir', default=None, help='log directory')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping')
    # Model parameters
    parser.add_argument('--char_emb_size', type=int, default=25, help='character embedding size')
    parser.add_argument('--word_emb_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--char_lstm_units', type=int, default=25, help='num of character lstm units')
    parser.add_argument('--word_lstm_units', type=int, default=100, help='num of word lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--no_char_feature', action='store_false', help='use char feature')
    parser.add_argument('--no_use_crf', action='store_false', help='use crf layer')
    parser.add_argument('--correct_spell', action='store_true',default=False, help='apply spell corretor')
    

    args = parser.parse_args()
    if args.save_dir :#
        save_dir =  os.path.abspath(os.path.join(MODELS_DIR,args.save_dir))
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        print(save_dir)
        args.weights_file   = os.path.join(save_dir,'weights.h5')
        args.params_file    = os.path.join(save_dir,'params.json')
        args.preprocessor_file = os.path.join(save_dir,'preprocessor.bin')
    pprint(args)
    main(args)
