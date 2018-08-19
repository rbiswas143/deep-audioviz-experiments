const expAvg = (old, curr, decayRate) => {
  curr = curr === undefined || isNaN(curr) ? 0 : curr;
  old = old === undefined || isNaN(old) ? 0 : old;
  return (curr * decayRate) + (old * (1 - decayRate));
};

export default class Mapper {

  constructor(viz, options) {
    this.viz = viz;
    this.active = true; // Set false to stop the mapper

    // Defaults
    this.options = Object.assign({
      mapInterval: 10,
      decayRate: 0.4,
      extrapol: true,
      useExpAvg: true
    }, options);
    this.updatePostProcessingOptions();
  }

  updatePostProcessingOptions() {
    // Exponentially Weighted Moving Average
    const expAvgSlider = document.getElementById('exp-avg');
    this.options.decayRate = 1 - parseFloat(expAvgSlider.value);

    // Linear Extrapolation
    const extrapolCheckbox = document.getElementById('extrapol-checkbox');
    this.options.extrapol = extrapolCheckbox.checked;
  }

  getReorderedParams() {
    // Extract animation params form the reorder params modal
    const orderedParams = [];
    const paramsList = document.querySelectorAll('.viz-param-item');
    for (let i = 0; i < paramsList.length; i++) {
      orderedParams.push(paramsList[i].getAttribute('viz-param-name'));
    }
    return orderedParams;
  }

  start(requestData, featuresData, audioManager) {
    // Global computations
    const hops = featuresData.dataset_config.mfcc_hops;
    const frames = featuresData.dataset_config.frames_per_segment;
    const sr = featuresData.dataset_config.sr;
    const featureDuration = hops / sr * frames;

    // Get params from modal
    const orderedParams = this.getReorderedParams();

    // Randomize feature order
    const featureOrder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    if (requestData.random) {
      featureOrder.sort(() => 0.5 - Math.random());
    }

    // Mapper loop
    const doMap = () => {
      if (!this.viz.vizParams.paused) { // Skip mapping when paused

        // Current feature
        const elapsedTime = audioManager.audio.currentTime;
        const currFrameIndex = Math.min(featuresData.encoding.length - 1, Math.floor(elapsedTime / featureDuration));
        const currFeature = featuresData.encoding[currFrameIndex];

        // Time elapsed in current feature
        const featureTime = Math.min(featureDuration, elapsedTime - (featureDuration * currFrameIndex));
        const featureRatio = featureTime / featureDuration;

        // Next feature (Use curr feature if extrapol is false or this is the last feature)
        const useNextFeature = this.options.extrapol && (currFrameIndex < featuresData.encoding.length - 1);
        const nextFeature = featuresData.encoding[currFrameIndex + (useNextFeature ? 1 : 0)];

        // Extrapolate
        const applyFeature = [];
        for (let i = 0; i < currFeature.length; i++) {
          const j = featureOrder[i];
          applyFeature[i] = (featureRatio * nextFeature[j]) + ((1 - featureRatio) * currFeature[j]);
        }

        // Map
        const newParams = {};
        for (let i = 0; i < orderedParams.length; i++) {
          newParams[orderedParams[i]] = applyFeature[i];
        }

        // Exponentially weighted Averaging
        if (this.options.useExpAvg) {
          Object.keys(newParams).forEach(key => {
            newParams[key] = expAvg(this.viz.animParams[key], newParams[key], this.options.decayRate);
          });
        }

        Object.assign(this.viz.animParams, newParams);
      }
      if (this.active) { // Stop mapping when inactive
        setTimeout(doMap, this.options.mapInterval);
      }
    };

    // Start mapper loop
    doMap();
  }
}
