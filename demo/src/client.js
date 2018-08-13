let api_url = null;
if (process.env.NODE_ENV === 'production') {
  api_url = 'http://rhinomaster.mypi.co:9916';
} else {
  api_url = 'http://localhost:7000'
}

export function validate_request_data(data) {
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

export function parse_request_data(currTrack) {

  // Form elements
  const modelSelect = document.getElementById("model-select");
  const ftMapSelect = document.getElementById("ft-map-select");
  const ftScaleSelect = document.getElementById("ft-scale-select");
  const ftScaleMethodSelect = document.getElementById("ft-scale-method-select");

  // Feature Mapping Options
  return {
    model: modelSelect.options[modelSelect.selectedIndex].getAttribute("model"),
    classifier_layer: modelSelect.options[modelSelect.selectedIndex].getAttribute("layer"),
    feature_mapping: ftMapSelect.options[ftMapSelect.selectedIndex].getAttribute("mapping"),
    random: ftMapSelect.options[ftMapSelect.selectedIndex].getAttribute("random") === "true",
    feature_scaling: ftScaleSelect.options[ftScaleSelect.selectedIndex].getAttribute("scaling"),
    scaling_method: ftScaleMethodSelect.options[ftScaleMethodSelect.selectedIndex].getAttribute("method"),
    track: currTrack.type === 'upload' ? currTrack.track : currTrack.track_id
  };
}

export function fetchData(data, onSuccess, onFailure) {
  const formData = new FormData();
  Object.keys(data).forEach(function (key) {
    formData.append(key, data[key]);
  });

  fetch(`${api_url}/fetchmap`, {
    method: "POST",
    body: formData
  })
    .then(response => response.json())
    .then(onSuccess, onFailure)
}

export function fetchTracks(onSuccess, onFailure) {

  fetch(`${api_url}/fetchtracks`, {
    method: "GET"
  })
    .then(response => response.json())
    .then(onSuccess, onFailure)
}
