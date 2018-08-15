import THREE from "../three";

import BaseViz from "./base-viz";

const vertexShader = require('./shaders/psy-vertex.glsl');
const fragmentShader = require('./shaders/psy-fragment.glsl');

export default class VizPsychedelic extends BaseViz {
  init() {

    this.vizParams = {
      camFov: 50,
      camNear: 0.1,
      camFar: 20000,
      camZ: 200,
      orbitalControls: true,
      brightMin: 0.3,
      brightMax: 0.8,
      saturationMin: 0.1,
      saturationMax: 0.9,
      colMin: 2,
      colMax: 8,
      paused: false
    };

    this.animParams = {
      hue: 0.1,
      hueVariation: 0.1,
      density: 0.9,
      displacement: 0.2,
      speed: 0.2,
      wavesX: 0.2,
      wavesY: 0.2,
      saturation: 0.2,
      brightness: 0.2,
      color: 0.2,
    };

    // Renderer
    this.renderer.alpha = true;

    this.timer = 0;

    // Plane
    this.createPlane();

  }

  static getVisualParamsInfo() {
    return [
      ['hue', 'Hue'],
      ['saturation', 'Saturation'],
      ['brightness', 'Brightness'],
      ['color', 'Colors'],
      ['hueVariation', 'Hue Variation'],
      ['density', 'Wave Density'],
      ['displacement', 'Wave Displacement'],
      ['speed', 'Wave Speed'],
      ['wavesX', 'Distortion along X Axis'],
      ['wavesY', 'Distortion along Y Axis']
    ]
  }

  createPlane() {
    this.material = new THREE.RawShaderMaterial({
      vertexShader: vertexShader,
      fragmentShader: fragmentShader,

      uniforms: {
        uTime: {type: 'f', value: 0},
        uHue: {type: 'f', value: .5},
        uHueVariation: {type: 'f', value: 1},
        uGradient: {type: 'f', value: 1},
        uDensity: {type: 'f', value: 1},
        uDisplacement: {type: 'f', value: 1},
        uMousePosition: {type: 'v2', value: new THREE.Vector2(0.5, 0.5)},
        uSaturation: {type: 'f', value: 0.6},
        uBrightness: {type: 'f', value: 0.5},
        uColorG: {type: 'f', value: 1},
        uColorB: {type: 'f', value: 1}
      }
    });
    this.planeGeometry = new THREE.PlaneGeometry(2, 2, 1, 1);
    this.plane = new THREE.Mesh(this.planeGeometry, this.material);
    this.scene.add(this.plane);
  }

  animate(time) {
    this.plane.material.uniforms.uHue.value = this.animParams.hue;
    this.plane.material.uniforms.uHueVariation.value = this.animParams.hueVariation;
    this.plane.material.uniforms.uDensity.value = this.animParams.density;
    this.plane.material.uniforms.uDisplacement.value = this.animParams.displacement;

    this.plane.material.uniforms.uSaturation.value = this.vizParams.saturationMin +
      ((this.vizParams.saturationMax - this.vizParams.saturationMin) * this.animParams.saturation);
    this.plane.material.uniforms.uBrightness.value = this.vizParams.brightMin +
      ((this.vizParams.brightMax - this.vizParams.brightMin) * this.animParams.brightness);

    this.plane.material.uniforms.uColorG.value = this.animParams.color;
    this.plane.material.uniforms.uColorB.value = 1 - this.animParams.color;

    this.timer += this.animParams.speed;
    this.plane.material.uniforms.uTime.value = this.timer;

    this.plane.material.uniforms.uMousePosition.value = new THREE.Vector2(this.animParams.wavesX, this.animParams.wavesY);
  }
}
