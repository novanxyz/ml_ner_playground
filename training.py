#!/home/grumpycat/anaconda3/bin/python
"""
Example from training to saving.
"""
import time

import argparse
import os
import json
from pprint import pprint
from datetime import datetime

from anago.utils import load_data_and_labels
from anago.models import BiLSTMCRF,load_model,save_model
from anago.preprocessing import IndexTransformer#, ELMoTransformer
from anago.trainer import Trainer
from anago.tagger import Tagger
from lang_util.spell import SpellCorrector
from keras.callbacks import ModelCheckpoint

spell = SpellCorrector()
def save_params_file(model,params_file):
    with open(params_file, 'w') as f:
        params = model.to_json()
        json.dump(json.loads(params), f, sort_keys=True, indent=4)


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def split_data(X,Y, ratio=0.3):    
    x_valid= []
    y_valid = [] 
    total = len( X )
    valid_num = int(ratio * total)
    for v in range(0,valid_num):
        x_valid.append( X.pop() )
        y_valid.append( Y.pop() )
        
    
    return X, Y, x_valid, y_valid

def main(args):
    print("Loading dataset... %s  at %s " % (args.train_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S") ))
    X, Y = load_data_and_labels(args.train_data)    
    # apply spell corrector befor training

    if args.correct_spell:
        print('applying spell corrector')
        start_time = time.time()
        X = [ spell.validate(" ".join(x)).split()  for x in X ]
        elapsed_time = time.time() - start_time
        print('applied spell corrector:' + str(elapsed_time))
        #save corrected file so no need to do this time consuming 
        corrected_file = args.train_data[:-4] + '.txt'
        with open(corrected_file,'w') as tf:
            for i,Q in enumerate(X):
                for w,word in enumerate(Q):
                    tf.write("%s\t%s\n" % (X[i][w],Y[i][w]))
                tf.write("\n")
       
    for idt,yt in enumerate(Y):
        for idx, yx in enumerate(yt):
            if " " in yx:
                y = yx.split(" ")
                y.sort()                
                yt[idx] = " ".join( y )

    x_train,y_train ,x_valid, y_valid = split_data(X,Y, int(args.ratio)/100)
        
    print(x_train[0],y_train[0])
    print(x_valid[0],y_valid[0])

    Y = set()
    for yt in y_train:
        for y in  (list(set(yt))):
            Y.add( y )    
    # pprint(Y)

    print('Transforming datasets...')
    p = IndexTransformer(use_char=args.no_char_feature)
    p.fit(x_train, y_train)

    print("Label size:",    str(p.label_size) , len(Y) )    
    print('Building a model.')

    if ( os.path.exists(args.weights_file) and os.path.exists(args.params_file) ):
        model = load_model(args.weights_file,args.params_file)        
#        loss = model.layers.pop().loss_function
    else:
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
    trained_path  = os.path.join( os.path.dirname(args.train_data) , 'trained')    
    last_epochs = 0  ## take biggest number of epochs trained to continue training
    epochs = args.epochs + last_epochs
    trainer = Trainer(model, preprocessor=p)

    checkpoints = [ ModelCheckpoint(args.save_dir + '/weights.h5', monitor='loss',mode='min',
                                verbose=0, save_best_only=True, save_weights_only=True,  period=1) ]
    trainer.train(x_train, y_train, x_valid, y_valid, 
                    batch_size=int(args.batch_size), 
                    epochs=epochs,initial_epoch=last_epochs, callbacks = checkpoints )




    print('Saving the model...:\n%s \n%s \%s' % (args.weights_file, args.params_file, args.preprocessor_file ) )    
    #save_model(model,args.weights_file , args.params_file)    
    save_params_file(model,args.params_file)
    p.save(args.preprocessor_file)    
    
    ##move trained data    
    if args.move_trained:
        print("Save Data for incremental")    
        trained_file= os.path.basename(args.train_data)     
        name, ext = trained_file.split(".")
        trained_file =  "%s-%d.%s" % (name,epochs, ext)    
        os.rename(args.train_data, os.path.realpath(os.path.join(trained_path,trained_file) ))    

    #model.score(x_valid,y_valid)    
    test = "kenapa bs hidrosefalus, dok anak pertama sya dibilang dokter kena hidrosefalus ada cairan yang berkumpul di bagian otaknya ini sebenarnya penyebabnya apa ya dok"
    test = "bayi bahayakah dok jika tidak membuat bayi selsaai menyusui? soalnya kan dok bayi sy asal di tepuk belakangnya dia tersentak dan tak mau tidur lagi"
    test = "dok apakah kehamilan anak pertama menyebabkan mual2 terus"
    test = "dok saya mau tanya bahan alami untuk obatin penyakit herpes apa"
    test = "ibu mertua sya biduran udah seminggu, obatnya apa ya dok?"
    if args.test:
        test = args.test

    test = spell.validate(test)
    tagger =  Tagger(model,p)
    res = tagger.analyze(test)
    pprint(res)

if __name__ == '__main__':
    DATA_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))
    MODELS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))
    parser      = argparse.ArgumentParser(description='Training NER Model')

    parser.add_argument('--train_data', default=os.path.join(DATA_DIR, 'train.txt'), help='training data')
    parser.add_argument('--valid_data', default=os.path.join(DATA_DIR, 'valid.txt'), help='validation data')
    parser.add_argument('--ratio', default=30,help="Ratio of Validation and Train data")
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--test', default=None)
    parser.add_argument('--move_trained', action='store_true',default=False, help='Move trained data')    
    parser.add_argument('--correct_spell', action='store_true',default=False, help='apply spell corrector')

    parser.add_argument('--weights_file', default=None, help='weights file')
    parser.add_argument('--params_file', default=None, help='parameter file')
    parser.add_argument('--preprocessor_file',default=None, help='preprocessor_file')
    # Training parameters
    parser.add_argument('--loss', default='categorical_crossentropy', help='loss')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--checkpoint_path', default=MODELS_DIR, help='checkpoint path')
    parser.add_argument('--log_dir', default=None, help='log directory')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping')
    # Model parameters
    parser.add_argument('--char_emb_size', type=int, default=25, help='character embedding size')
    parser.add_argument('--word_emb_size', type=int, default=150, help='word embedding size')
    parser.add_argument('--char_lstm_units', type=int, default=25, help='num of character lstm units')
    parser.add_argument('--word_lstm_units', type=int, default=150, help='num of word lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--no_char_feature', action='store_false', help='use char feature')
    parser.add_argument('--no_use_crf', action='store_false', help='use crf layer')
    

    args = parser.parse_args()

    if not args.save_dir and args.ratio:
        args.save_dir =  os.path.abspath(os.path.join(MODELS_DIR, 'ratio' + args.ratio ))

    if args.save_dir :#
        save_dir =  os.path.abspath(os.path.join(MODELS_DIR,args.save_dir))
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        args.weights_file   = os.path.join(save_dir,'weights.h5')
        args.params_file    = os.path.join(save_dir,'params.json')
        args.preprocessor_file = os.path.join(save_dir,'preprocessor.bin')
    pprint(args)
    main(args)
