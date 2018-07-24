import data_processor as dp
import train
import models
import mapping_utils

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch

# frontend params: deterministic/random, exp avg


class ServerConfig(dp.BaseConfig):

    def __init__(self):
        self.secret = 'saodoasmdom9908128euh1dn'
        self.port = 7000
        self.debug = True
        self.upload_dir = '/tmp/deepviz'
        self.allowed_extensions = ['mp3', 'wav']


def start_server(server_config):

    app = Flask(__name__)
    app.secret_key = server_config.secret
    app.config['server_config'] = server_config

    @app.route('/fetchmap', methods=['GET', 'POST'])
    def fetch_map():
        # if not request.method == 'POST':
        #     raise Exception('Invalid request method: {}'.format(request.method))

        if request.method == 'GET':
            return '''
                    <!doctype html>
                    <title>Upload new File</title>
                    <h1>Upload new File</h1>
                    <form method=post enctype=multipart/form-data>
                      Model <input type=text name=model><br>
                      Mapping <input type=text name=feature_mapping><br>
                      Scaling <input type=text name=feature_scaling><br>
                      Softmax <input type=text name=kmeans_softmax><br>
                      Layer <input type=text name=classifier_layer><br>
                      Track <input type=file name=track><br>
                      <input type=submit value=Upload>
                    </form>
                    '''

        # Request Config
        request_config = mapping_utils.MappingConfig()
        request_config.model = request.form['model']
        request_config.train_config_path = models.trained_model_configs[request_config.model]
        request_config.feature_mapping = request.form['feature_mapping']
        request_config.feature_scaling = request.form['feature_scaling']
        if 'kmeans_softmax' in request.form:
            request_config.kmeans_softmax = request.form['kmeans_softmax'].lower() == 'true'
        if 'classifier_layer' in request.form:
            request_config.classifier_layer = request.form['classifier_layer']

        # File
        if 'track' not in request.files:
            raise Exception('Audio file is not available in the request')
        track = request.files['track']
        if track.filename == '':
            raise Exception('No selected audio file')
        server_config = app.config['server_config']
        if not (track and '.' in track.filename and track.filename
                .rsplit('.', 1)[1].lower() in server_config.allowed_extensions):
            raise Exception('Invalid file')
        filename = secure_filename(track.filename)
        os.makedirs(server_config.upload_dir, exist_ok=True)
        track_path = os.path.join(server_config.upload_dir, filename)
        track.save(track_path)

        try:
            train_config = train.TrainingConfig.load_from_file(request_config.train_config_path)
            dataset_config = dp.DataPrepConfig.load_from_dataset(train_config.dataset_path)
            dataset_mode = dp.read_h5_attrib('mode', dataset_config.get_dataset_path())

            # Model
            cuda = torch.cuda.is_available()
            model = train_config.get_by_model_key(cuda)
            model.load_state(train_config.get_model_path('state_best'))

            # Process track
            partition = mapping_utils.generate_partition(track_path, dataset_mode, dataset_config)
            batch = dp.PartitionBatchGenerator(partition, train_config.batch_size, mode='track')

            # Encode and Map
            enc = mapping_utils.encode(model, batch, train_config, request_config)
            enc = mapping_utils.get_mapping(enc, request_config, train_config)
            enc = mapping_utils.get_enc_scaled(enc, request_config.feature_scaling)
        finally:
            os.unlink(track_path)

        return jsonify({
            'train_config': train_config.get_dict(),
            'dataset_mode': dataset_mode,
            'dataset_config': dataset_config.get_dict(),
            'encoding': enc.tolist()
        })

    print('Starting DeepViz server at Port:{}'.format(server_config.port))
    app.run(None, server_config.port, debug=server_config.debug)


if __name__ == '__main__':
    start_server(ServerConfig())