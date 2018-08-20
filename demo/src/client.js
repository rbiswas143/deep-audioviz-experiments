// Prod api has the same origin
export let api_url = '/api';
if (process.env.mode === 'development') {
  api_url = `${process.env.api_host}:${process.env.api_port}`
}

const useLocalStorage = false;

export default class Client {
  constructor() {
    // Form elements
    this.modelSelect = document.getElementById("model-select");
    this.ftMapSelect = document.getElementById("ft-map-select");
    this.ftScaleSelect = document.getElementById("ft-scale-select");
    this.ftScaleMethodSelect = document.getElementById("ft-scale-method-select");

    // Cache
    this.cache = {}
  }

  validate_request_data(data) {
    let valid = true;

    // No empty fields
    ['model', 'feature_mapping', 'feature_scaling', 'scaling_method', 'track'].forEach(key => {
      if (!data[key]) {
        valid = false;
      }
    });

    // Disallowed combinations
    if (data.feature_mapping === 'raw' && (data.model === 'alexnet' ||
      (typeof(data.model) === 'string' && data.model.startsWith('vgg')))) {
      alert('Sorry, Genre Classifiers cannot be used with Deterministic and Random Mapping.');
      valid = false;
    }

    return valid;
  }

  parse_request_data(currTrack) {

    // Feature Mapping Options
    return {
      model: this.modelSelect.options[this.modelSelect.selectedIndex].getAttribute("model"),
      classifier_layer: this.modelSelect.options[this.modelSelect.selectedIndex].getAttribute("layer"),
      feature_mapping: this.ftMapSelect.options[this.ftMapSelect.selectedIndex].getAttribute("mapping"),
      random: this.ftMapSelect.options[this.ftMapSelect.selectedIndex].getAttribute("random") === "true",
      feature_scaling: this.ftScaleSelect.options[this.ftScaleSelect.selectedIndex].getAttribute("scaling"),
      scaling_method: this.ftScaleMethodSelect.options[this.ftScaleMethodSelect.selectedIndex].getAttribute("method"),
      track: currTrack.type === 'upload' ? currTrack.track : currTrack.track_id
    };
  }

  _getCacheKey(data) {
    let vals = [];
    Object.keys(data).forEach(key => {
      if (key !== 'track') vals.push(data[key]);
    });
    if (data.track instanceof File) {
      vals = vals.concat(['name', 'lastModified', 'size', 'type'].map(attrib => data.track[attrib]));
    } else {
      vals.push(data.track);
    }
    const cache_key = vals.join('|');
    console.log('Cache Key', cache_key);
    return cache_key;
  }

  _getCached(cacheKey) {
    let cached = null;
    if (cacheKey in this.cache) {
      cached = this.cache[cacheKey];
    } else if (useLocalStorage) {
      cached = JSON.parse(localStorage.getItem(cacheKey));
    }
    return cached;
  }

  _saveToCache(cacheKey, data) {
    this.cache[cacheKey] = data;
    if (useLocalStorage) {
      localStorage.setItem(cacheKey, JSON.stringify(data));
    }
  }

  fetchData(data, onSuccess, onFailure) {
    const formData = new FormData();
    Object.keys(data).forEach(function (key) {
      formData.append(key, data[key]);
    });

    // Fetch cached
    const cacheKey = this._getCacheKey(data);
    const cachedData = this._getCached(cacheKey);
    if (cachedData) {
      new Promise(resolve => {
        console.log('Returning cached data');
        resolve(cachedData);
      }).then(onSuccess);
      return;
    }

    fetch(`${api_url}/fetchmap`, {
      method: "POST",
      body: formData
    })
      .then(response => response.json())
      .then(featuresData => {
        // Cache
        console.log('Caching new data');
        this._saveToCache(cacheKey, featuresData);
        onSuccess(featuresData);
      }, onFailure)
  }

}
