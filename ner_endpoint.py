#!/home/grumpycat/anaconda3/bin/python

from anago.tagger import Tagger
from anago.models import BiLSTMCRF, load_model
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from lang_util.spell import SpellCorrector

import traceback
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api

#models
weights_file    = ".models/weights.h5"
params_file     = ".models/params.json"
preprocessor_file = ".models/preprocess.bin"

AUTH_UNIQUE_TOKEN = 'Dn5JBLiUi6Afq33j9KcyZBZycvCIyrNrtwaFM0GCBFdkJYnhxo'

model = load_model(weights_file, params_file)
it = IndexTransformer.load(preprocessor_file)
tagger = Tagger(model, preprocessor=it)
spell = SpellCorrector()

app = Flask(__name__)
api = Api(app)

class Tagger(Resource):
    def post(self):
        try:
            # app.logger.info(request.headers)
            if 'Auth-Unique-Token' not in request.headers.keys() or AUTH_UNIQUE_TOKEN != request.headers['Auth-Unique-Token']:
                return 'INVALID TOKEN', 404

            json = request.get_json(force=True)

            test = spell.validate(json['data']['body'])
            res = tagger.analyze(test)
            app.logger.info('[TAGGER] '+ content +' -> '+ str(res))
            return make_response(jsonify(tag=res), 200)

        except ex:
            app.logger.info(traceback.format_exc())
            return make_response(jsonify(tag='None'), 200)



class Trainer(Resource):
    def post(self)
        try:

            if 'Auth-Unique-Token' not in request.headers.keys() or AUTH_UNIQUE_TOKEN != request.headers['Auth-Unique-Token']:
                return 'INVALID TOKEN', 404

            file = request.files['file']

            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            
            
        except ex:
            app.logger.info(traceback.format_exc())
            return make_response(jsonify(tag='None'), 500)


