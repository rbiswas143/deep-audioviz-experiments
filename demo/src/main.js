import * as dat from 'dat.gui';
import 'bootstrap';
import '../node_modules/bootstrap/dist/css/bootstrap.css';
import Detector from 'three/examples/js/Detector';

import AudioManager from './audio-manager';
import * as Fullscreen from './fullscreen';
import FMATrackSelector from './fma-selector';
import VizBoxMessages from './messages';
import * as client from './client';
import {vizMap} from './viz/all';
import Mapper from './viz/mapper';
import VizParamsReordering from './reorder-params';
import './styles.css';
import logo from '../assets/logo.png';

const enableDatGUI = false;

class Main {
  constructor() {

    // Global
    this.audioManager = new AudioManager();
    this.vizBoxMessages = new VizBoxMessages();
    // init after user selection
    this.viz = null;
    this.mapper = null;
    this.currTrack = null;

    // UI Elements
    this.vizBox = document.getElementById('viz-box');
    this.vizContainer = document.getElementById('viz-container');
    this.menuBox = document.getElementById('menu-box');
    this.selectedTrackInfo = document.getElementById('selected-track-info');
    this.vizSelect = document.getElementById('viz-select');

    // Init
    this.initUI();
    this.initFullScreen();
    this.initFMATrackSelector();
    this.initTrackUploader();
    this.initTrackControl();
    this.initPostProcessing();
    this.initVizParamsReordering();
    this.initGoButton();

  }

  initUI() {
    // Load logo
    document.getElementById('logo').setAttribute('src', logo);

    // Show UI
    document.querySelector('body').style.visibility = 'visible';
  }

  initFullScreen() {

    // Bind keys
    Fullscreen.bindKey({
      charCode: 'F'.charCodeAt(0),
      dblclick: true,
      element: this.vizBox,
    });

    // On fullscreen change event handling
    Fullscreen.changeSuccessCallback(() => {
      const activated = Fullscreen.activated();
      if (activated) {
        this.vizBox.classList.add('viz-fullscreen');
        this.menuBox.classList.add('viz-fullscreen');
      } else {
        this.vizBox.classList.remove('viz-fullscreen');
        this.menuBox.classList.remove('viz-fullscreen');
      }
      this.viz && this.viz.resize();
    });

    // Track control
    document.getElementById('viz-control-fullscreen').onclick = () => {
      Fullscreen.toggle(this.vizBox);
    }
  }

  initGoButton() {

    document.getElementById('go-btn').onclick = () => {

      if (this.currTrack) {

        this.destroy();

        const requestData = client.parse_request_data(this.currTrack);
        if (!client.validate_request_data(requestData)) return;

        this.vizBoxMessages.setLoading();
        client.fetchData(
          requestData,
          featuresData => {
            this.vizBoxMessages.hide();
            this.start(requestData, featuresData);
          },
          error => {
            this.vizBoxMessages.setError();
            console.log('Error:', error);
          }
        );
      } else {
        alert('Choose a track first.');
      }
    };
  }

  initFMATrackSelector() {
    const fmaTrackSelector = new FMATrackSelector();
    fmaTrackSelector.fmaLoaderButton[0].onclick = () => {
      this.currTrack = fmaTrackSelector.getCurrFMATrack();
      this.selectedTrackInfo.innerHTML = fmaTrackSelector.getTrackDisplayName(this.currTrack);
    };
  }

  initTrackUploader() {
    const trackLoader = document.getElementById('track-loader');
    trackLoader.onchange = () => {
      if (trackLoader.files.length > 0) {
        this.currTrack = {
          track: trackLoader.files[0],
          title: trackLoader.files[0].name,
          src: URL.createObjectURL(trackLoader.files[0]),
          type: 'upload',
        };
        this.selectedTrackInfo.innerHTML = `[UPLOAD] ${this.currTrack.title}`;
      }
    };
  }

  initTrackControl() {
    this.audioManager.onPause = () => {
      this.viz && (this.viz.vizParams.paused = true);
    };

    this.audioManager.onPlay = () => {
      this.viz && (this.viz.vizParams.paused = false);
    };

    this.audioManager.onStop = () => {

    }

  }

  initPostProcessing() {
    // Exponentially Weighted Moving Average
    const expAvgSlider = document.getElementById('exp-avg');
    expAvgSlider.onchange = () => {
      this.mapper && this.mapper.updatePostProcessingOptions();
    };

    // Linear Extrapolation
    const extrapolCheckbox = document.getElementById('extrapol-checkbox');
    extrapolCheckbox.onchange = () => {
      this.mapper && this.mapper.updatePostProcessingOptions();
    };
  }

  initVizParamsReordering() {
    new VizParamsReordering();
  }

  start(requestData, featuresData) {

    // Detect WEBGL
    if (!Detector.webgl) {
      this.vizBoxMessages.updateMessage(true, Detector.getWebGLErrorMessage().outerHTML);
      return;
    }

    // Viz Code
    const vizCode = this.vizSelect.options[this.vizSelect.selectedIndex].value;
    if (!vizCode) {
      console.log('vizCode is not available');
      this.vizBoxMessages.setError();
    }

    // Set up Viz Box
    this.vizBox.classList.add('viz-active');

    // Viz and Mapper
    this.viz = new vizMap[vizCode](this.vizContainer);
    this.mapper = new Mapper(this.viz);

    if (enableDatGUI) {
      let gui = new dat.GUI();
      for (let key in this.viz.animParams) {
        if (this.viz.animParams.hasOwnProperty(key)) {
          gui.add(this.viz.animParams, key).listen();
        }
      }
    }

    // Start track, mapper and visualization
    this.audioManager.loadTrack(this.currTrack.src, () => {
      this.viz.start();
      this.mapper.start(requestData, featuresData, this.audioManager);
      this.audioManager.activate();
      this.audioManager.play();
    });
  }

  destroy() {
    this.audioManager && this.audioManager.pause();
    this.audioManager.deactivate();
    this.vizBox.classList.remove('viz-active');
    this.viz && this.viz.destroy();
    this.viz = null;
    this.mapper && (this.mapper.active = false);
    this.mapper = null;
  }

}

window.onload = () => new Main();
