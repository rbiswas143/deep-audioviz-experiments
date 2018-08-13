const expAvg = (old, curr, decayRate) => {
  curr = curr === undefined || isNaN(curr) ? 0 : curr;
  old = old === undefined || isNaN(old) ? 0 : old;
  return (curr * decayRate) + (old * (1 - decayRate));
};

export default class Mapper {

  constructor(viz, options) {
    this.viz = viz;
    this.active = true;
    this.options = Object.assign({
      mapInterval: 20,
      decayRate: 0.4,
      extrapol: true,
      useExpAvg: true
    }, options);
    this.updatePostProcessingOptions();
  }

  updatePostProcessingOptions() {
    // Exponentially Weighted Moving Average
    const expAvgSlider = document.getElementById('exp-avg');
    this.options.decayRate = parseFloat(expAvgSlider.value);

    // Linear Extrapolation
    const extrapolCheckbox = document.getElementById('extrapol-checkbox');
    this.options.extrapol = extrapolCheckbox.checked;
  }

  getOrderedParams() {
    const orderedParams = [];
    const paramsList = document.querySelectorAll('.viz-param-item');
    for (let i = 0; i < paramsList.length; i++) {
      orderedParams.push(paramsList[i].getAttribute('viz-param-name'));
    }
    return orderedParams;
  }

  start(requestData, featuresData, audioManager) {
    const hops = featuresData.dataset_config.mfcc_hops;
    const frames = featuresData.dataset_config.frames_per_segment;
    const sr = featuresData.dataset_config.sr;
    const featureDuration = hops / sr * frames;

    const orderedParams = this.getOrderedParams();
    const featureOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    if (requestData.random) {
      featureOrder.sort(() => 0.5 - Math.random());
    }

    const doMap = () => {
      if (!this.viz.vizParams.paused) {
        const elapsedTime = audioManager.audio.currentTime;
        const currFrameIndex = Math.min(featuresData.encoding.length - 1, Math.floor(elapsedTime / featureDuration));
        const featureTime = Math.min(featureDuration, elapsedTime - (featureDuration * currFrameIndex));
        const featureRatio = featureTime / featureDuration;

        const currFeature = featuresData.encoding[currFrameIndex];
        const useNextFeature = this.options.extrapol && (currFrameIndex < featuresData.encoding.length - 1);
        const nextFeature = featuresData.encoding[currFrameIndex + (useNextFeature ? 1 : 0)];
        const applyFeature = [];
        for (let i = 0; i < currFeature.length; i++) {
          const j = featureOrder[i];
          applyFeature[i] = (featureRatio * nextFeature[j]) + ((1 - featureRatio) * currFeature[j]);
        }
        const newParams = this.map(applyFeature, orderedParams);
        if (this.options.useExpAvg) {
          Object.keys(newParams).forEach(key => {
            newParams[key] = expAvg(this.viz.animParams[key], newParams[key], this.options.decayRate);
          });
        }
        Object.assign(this.viz.animParams, newParams);
      }
      if (this.active) {
        setTimeout(doMap, this.options.mapInterval);
      }
    };

    doMap();
  }

  map(dnFeatures, params) {
    const mapped = {};
    for (let i = 0; i < params.length; i++) {
      mapped[params[i]] = dnFeatures[i];
    }
    return mapped;
  }
}
