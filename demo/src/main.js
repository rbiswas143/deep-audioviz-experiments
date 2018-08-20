import * as dat from 'dat.gui';
import 'bootstrap';
import '../node_modules/bootstrap/dist/css/bootstrap.css';
import Detector from 'three/examples/js/Detector';

import AudioManager from './audio-manager';
import * as Fullscreen from './fullscreen';
import FMATrackSelector from './fma-selector';
import VizBoxMessages from './messages';
import Client from './client';
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
    this.client = new Client();

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

    // Add remove classes on fulscreen change
    Fullscreen.changeSuccessCallback(() => {
      const activated = Fullscreen.activated();
      if (activated) {
        this.vizBox.classList.add('viz-fullscreen');
        this.menuBox.classList.add('viz-fullscreen');
      } else {
        this.vizBox.classList.remove('viz-fullscreen');
        this.menuBox.classList.remove('viz-fullscreen');
      }
    });

    // Track control - fullscreen button
    document.getElementById('viz-control-fullscreen').onclick = () => {
      Fullscreen.toggle(this.vizBox);
    }
  }

  initGoButton() {

    const goBtn = document.getElementById('go-btn');
    goBtn.onclick = () => {

      if (this.currTrack) { // Track is selected, proceed to load features

        // Validate form and get features from server
        const requestData = this.client.parse_request_data(this.currTrack);
        if (!this.client.validate_request_data(requestData)) return;

        // Destroy everything
        this.destroy();

        goBtn.disabled = true;
        this.vizBoxMessages.setLoading();
        this.client.fetchData(
          requestData,
          featuresData => {
            // Success: proceed to show viz
            this.vizBoxMessages.hide();
            this.start(requestData, featuresData);
          },
          error => {
            // Failure: show error message
            this.vizBoxMessages.setError();
            console.log('Error:', error);
          }
        ).finally(() => goBtn.disabled = false);
      } else {// Track is not selected
        alert('Choose a track first.');
      }
    };
  }

  initFMATrackSelector() {
    // On button click, update current track related info
    const fmaTrackSelector = new FMATrackSelector();
    fmaTrackSelector.fmaLoaderButton[0].onclick = () => {
      this.currTrack = fmaTrackSelector.getCurrFMATrack();
      this.selectedTrackInfo.innerHTML = fmaTrackSelector.getTrackDisplayName(this.currTrack);
    };
  }

  initTrackUploader() {
    // On file upload, upload current track related info
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
    // Play and pause viz along with the track
    this.audioManager.onPause = () => {
      this.viz && (this.viz.vizParams.paused = true);
    };
    this.audioManager.onPlay = () => {
      this.viz && (this.viz.vizParams.paused = false);
    };
  }

  initPostProcessing() {
    // Exponentially Weighted Moving Average (update mapper)
    const expAvgSlider = document.getElementById('exp-avg');
    expAvgSlider.onchange = () => {
      this.mapper && this.mapper.updatePostProcessingOptions();
    };

    // Linear Extrapolation (update mapper)
    const extrapolCheckbox = document.getElementById('extrapol-checkbox');
    extrapolCheckbox.onchange = () => {
      this.mapper && this.mapper.updatePostProcessingOptions();
    };
  }

  initVizParamsReordering() {
    new VizParamsReordering();
  }

  start(requestData, featuresData) {

    // Detect webgl
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

    // Add class to viz box to set it up
    this.vizBox.classList.add('viz-active');

    // Viz and Mapper
    this.viz = new vizMap[vizCode](this.vizContainer);
    this.mapper = new Mapper(this.viz);

    // Enable dat.gui [NOT USED]
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
    // Deactivate audio manager
    this.audioManager.pause();
    this.audioManager.deactivate();

    // Reset viz box
    this.vizBox.classList.remove('viz-active');

    // Destroy viz
    this.viz && this.viz.destroy();
    this.viz = null;

    // Destroy mapper
    this.mapper && (this.mapper.active = false);
    this.mapper = null;
  }

}

// Begin initialization on load
window.onload = () => new Main();
