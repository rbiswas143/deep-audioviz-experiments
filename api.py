"""Server exposing endpoints for uploading and downloading tracks, extracting their encodings and returning the
mapped visual parameters
"""

import data_processor as dp
import train
import models
import mapping_utils
import commons
import fma_utils

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import torch
from geolite2 import geolite2

# Logging
logfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', 'deepviz-requests.log')
os.makedirs(os.path.dirname(logfile), exist_ok=True)
logger = logging.getLogger('deepviz')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(logfile, maxBytes=1024 * 1024, backupCount=100)  # 1MB x 100 files
formatter = logging.Formatter('%(asctime)s :: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ServerConfig(commons.BaseConfig):
    """Server Initialization Config
    The defaults defined here should be overridden using a json config file
    """

    def __init__(self):
        self.secret = 'saodoasmdom9908128euh1dn'
        self.port = 7000
        self.debug = False
        self.upload_dir = '/tmp/deepviz'
        self.allowed_extensions = ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']


# Load server config
server_config = ServerConfig.load_from_file('private/server_config.json')

# Server configuration
app = Flask(__name__)
CORS(app)
app.secret_key = server_config.secret
app.config['server_config'] = server_config

# Geolite
geo_reader = geolite2.reader()

# Global variables
request_count = 0


@app.route('/fetchmap', methods=['POST'])
def fetch_map():
    """Parse form data and track, extract features and return mapped visual params"""

    # Parse Request Config
    request_config = mapping_utils.MappingConfig()
    request_config.model = request.form['model']
    request_config.train_config_path = models.trained_model_configs[request_config.model]
    request_config.feature_mapping = request.form['feature_mapping']
    request_config.feature_scaling = request.form['feature_scaling']
    if 'classifier_layer' in request.form:
        request_config.classifier_layer = request.form['classifier_layer']
    logger.info('Request Config: %s', json.dumps(request_config.get_dict(), indent=2, sort_keys=True))

    # Configs
    train_config = train.TrainingConfig.load_from_file(request_config.train_config_path)
    dataset_config = dp.DataPrepConfig.load_from_dataset(train_config.dataset_path)
    dataset_mode = dp.read_h5_attrib('mode', dataset_config.get_dataset_path())

    # Save uploaded file or get FMA track path
    track_type = 'upload'
    if 'track' not in request.files:
        if 'track' in request.form:
            track_path = fma_utils.get_audio_path('datasets/fma/fma_small', int(request.form['track']))
            track_type = 'fma'
        else:
            logger.error('Audio file is not available in the request')
            raise Exception('Audio file is not available in the request')
    else:
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
    logger.info('Track type : %s\tTrack path : %s', track_type, track_path)

    try:
        # Model
        cuda = torch.cuda.is_available()
        model = train_config.get_by_model_key(cuda)
        model.load_state(train_config.get_model_path('state_best'))

        # Process track
        partition = mapping_utils.generate_partition(track_path, dataset_mode, dataset_config)
        batch = dp.PartitionBatchGenerator(partition, train_config.batch_size, mode='track')

        # Encode and Map
        enc = mapping_utils.encode(model, batch, train_config, request_config)
        enc = mapping_utils.map_and_scale(enc, request_config, train_config)
    finally:
        # Delete uploaded track
        if track_type == 'upload':
            os.unlink(track_path)

    # Compile and send
    return jsonify({
        'train_config': train_config.get_dict(),
        'dataset_mode': dataset_mode,
        'dataset_config': dataset_config.get_dict(),
        'encoding': enc.tolist()
    })


@app.route('/fetchtracks', methods=['GET'])
def fetch_tracks():
    """Fetch metadata for FMA Small"""

    tracks = commons.get_fma_meta("datasets/fma/fma_metadata", 'small')
    return jsonify(list(zip(*[
        tracks.index.tolist(),
        tracks['track', 'title'].tolist(),
        tracks['artist', 'name'].tolist(),
        tracks['track', 'genre_top'].tolist()
    ])))


@app.route('/downloadtrack/<path:track_id>', methods=['GET'])
def download_fma_track(track_id):
    """Download a track from FMA small"""

    track_path = fma_utils.get_audio_path('datasets/fma/fma_small', int(track_id))
    return send_file(track_path)


@app.after_request
def log_request(resp):
    """Log request details after each request"""

    request_data = {
        'endpoint': request.endpoint,
        'host_url': request.host_url,
        'referrer': request.referrer,
        'method': request.method,
        'remote_addr': request.remote_addr,
        'user_agent': str(request.user_agent)
    }
    logger.info('Request Data: %s', json.dumps(request_data, indent=2, sort_keys=True))

    try:
        ip = request.remote_addr
        geo_data = geo_reader.get(ip)
        logger.info('Geo Data: %s', json.dumps(geo_data, indent=2, sort_keys=True))
    except:
        logger.exception('Failed to log geo data')

    logger.info('Response status: %s', resp.status)

    global request_count
    request_count += 1
    print('Requests Count: {}'.format(request_count))

    return resp


# Start dev server
if __name__ == '__main__':
    logger.info('Starting DeepViz server at Port:{}'.format(server_config.port))
    app.run('0.0.0.0', server_config.port, debug=server_config.debug)
