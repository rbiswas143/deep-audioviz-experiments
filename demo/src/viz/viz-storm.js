import THREE from "../three";

import BaseViz from "./base-viz";
import Tornado from "./comps/tornado";
import Rain from "./comps/rain";
import {scaleToRange} from "./utils";

export default class VizStorm extends BaseViz {
  init() {

    this.vizParams = {
      camFov: 60,
      camNear: 0.1,
      camFar: 100000,
      camX: 3,
      camY: -1,
      camZ: -2,
      speedMin: 0,
      speedMax: 0.5,
      cloudResolutionMin: 20,
      cloudResolutionMax: 70,
      cloudNumBlobsMin: 5,
      cloudNumBlobsMax: 20,
      cloudSubtractMin: 20,
      cloudSubtractMax: 100,
      cloudIsolationMin: 0,
      cloudIsolationMax: 80,
      lightningIntervalMin: 0.4,
      lightningIntervalMax: 3,
      lightningDurationMin: 0.1,
      lightningDurationMax: 3,
      tornadoRingMin: 0,
      tornadoRingMax: 0.7,
      tornadoLatheMin: 0.01,
      tornadoLatheMax: 0.07,
      numRainDrops: 200,
      rainMin: -7,
      rainMax: -4,
      nightMin: -0.4,
      nightMax: 0.1,
      orbitalControls: true,
      orbitalControlsTarget: [0, 1, 0],
      paused: false,
      time: 0
    };

    this.animParams = {
      speed: 0.5,
      cloudResolution: 0.5,
      cloudNumBlobs: 0.5,
      cloudSize: 0.5,
      cloudIsolation: 0.5,
      lightningDuration: 0.5,
      lightningInterval: 0.5,
      tornado: 0.5,
      rain: 0.5,
      night: 0.5,
    };

    // Fog
    this.scene.background = new THREE.Color(0x716c7c);
    this.scene.fog = new THREE.Fog(this.scene.background, 1, 5);
    this.bg = this.scene.background.getHSL({});

    // Lights
    this.aLight = new THREE.AmbientLight(0x2e2833);
    this.scene.add(this.aLight);

    // Sun or Moon
    this.dLight = new THREE.DirectionalLight(0xffffff, 0.5);
    this.dLight.position.set(1, 1, 1);
    this.scene.add(this.dLight);

    // Lightning
    this.pLight = new THREE.PointLight(0xffff00, 0);
    this.pLight.position.y = 1;
    this.scene.add(this.pLight);

    this.clock = new THREE.Clock();
    this.lastLightningTime = 0;


    // Components
    this.comps = [
      new Tornado(this.vizParams, this.animParams),
      new Rain(this.vizParams, this.animParams)
    ];
    this.comps.forEach(comp => {
      this.scene.add(comp);
    });

    this.createClouds();


  }

  createClouds() {
    this.cloudsMat = new THREE.MeshPhongMaterial({
      color: 0xd9edfd,
      specular: 0x111111,
      shininess: 1
    });
    this.clouds = new THREE.MarchingCubes(32, this.cloudsMat, false, false);
    this.clouds.isolation = 80;
    this.clouds.position.set(0, 2, 0);
    this.clouds.scale.setScalar(3);
    this.scene.add(this.clouds);
  }

  static getVisualParamsInfo() {
    return [
      ['speed', 'Cloud Speed'],
      ['cloudResolution', 'Cloud Resolution'],
      ['cloudNumBlobs', 'Number of Cloud Blobs'],
      ['cloudSize', 'Cloud Size'],
      ['cloudIsolation', 'Cloud Separation'],
      ['lightningDuration', 'Lightning Duration'],
      ['lightningInterval', 'Lightning Interval'],
      ['tornado', 'Tornado Intensity'],
      ['rain', 'Rain Speed'],
      ['night', 'Darkness']
    ]
  }

  animate(time) {
    const speed = scaleToRange(this.animParams.speed, this.vizParams.speedMin, this.vizParams.speedMax);
    this.vizParams.time += this.clock.getDelta() * speed;

    // Night
    const capL = l => Math.max(Math.min(l, 1), 0);
    const nightVal = scaleToRange(1 - this.animParams.night, this.vizParams.nightMin, this.vizParams.nightMax);

    // lightning flicker
    const lightningDuration = scaleToRange(this.animParams.lightningDuration,
      this.vizParams.lightningDurationMin, this.vizParams.lightningDurationMax);
    const lightningInterval = scaleToRange(this.animParams.lightningInterval,
      this.vizParams.lightningIntervalMin, this.vizParams.lightningIntervalMax);
    const elapsedTime = Math.round(this.clock.elapsedTime);
    if (elapsedTime - this.lastLightningTime > lightningInterval && Math.random() > 0.8) {
      if (elapsedTime - this.lastLightningTime > lightningInterval + lightningDuration) {
        this.lastLightningTime = elapsedTime;
      }
      this.pLight.intensity = 0.6;
      this.scene.background.setHSL(this.bg.h, this.bg.s, capL(this.bg.l + nightVal + 0.4));
    } else {
      this.pLight.intensity = 0;
      this.scene.background.setHSL(this.bg.h, this.bg.s, capL(this.bg.l + nightVal));
    }
    this.scene.fog.color.set(this.scene.background);

    this.animateClouds();

    this.comps.forEach(comp => {
      comp.animate && comp.animate(time);
    });
  }

  animateClouds() {
    // Cloud Resloution
    const cloudResolution = Math.floor(scaleToRange(this.animParams.cloudResolution,
      this.vizParams.cloudResolutionMin, this.vizParams.cloudResolutionMax));
    this.clouds.init(Math.floor(cloudResolution));

    // Cloud Isolation
    this.clouds.isolation = Math.floor(scaleToRange(this.animParams.cloudIsolation,
      this.vizParams.cloudIsolationMin, this.vizParams.cloudIsolationMax));

    this.clouds.reset();

    // Cloud Num Blobs
    const numBlobs = Math.floor(scaleToRange(this.animParams.cloudNumBlobs,
      this.vizParams.cloudNumBlobsMin, this.vizParams.cloudNumBlobsMax));
    const strength = 1.2 / ((Math.sqrt(numBlobs) - 1) / 4 + 1);

    // Cloud Size
    const cloudSubtract = Math.floor(scaleToRange(1 - this.animParams.cloudSize,
      this.vizParams.cloudSubtractMin, this.vizParams.cloudSubtractMax));

    for (let i = 0; i < numBlobs; i++) {
      let ballx = Math.sin(i + 1.26 * this.vizParams.time * (1.03 + 0.5 * Math.cos(0.21 * i))) * 0.27 + 0.5;
      let bally = Math.sin(i + 0.2 * this.vizParams.time) * .08 + 0.5;
      let ballz = Math.cos(i + 1.32 * this.vizParams.time * 0.1 * Math.sin((0.92 + 0.53 * i))) * 0.27 + 0.5;
      this.clouds.addBall(ballx, bally, ballz, strength, cloudSubtract);
    }
  }

  destroy() {
    this.comps.forEach(comp => {
      this.scene.remove(comp);
    });
    super.destroy();
  }
}
