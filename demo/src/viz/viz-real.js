import THREE from "../three";

import Spheres from "./comps/spheres";
import Sky from "./comps/sky";
import BaseViz from "./base-viz";

export default class VizReal extends BaseViz {
  init() {
    this.vizParams = {
      camFov: 75,
      camNear: 0.1,
      camFar: 1000,
      camRad: 300,
      orbitalControls: true,
      skyRad: 500,
      skyShininess: 0,
      skySpecular: '#000000',
      sphRadMax: 70,
      sphPosMax: 300,
      sphPosMin: 0,
      camFreqFactor: 3,
      camHeightMax: 200,
      shadow: true,
      skyTexture: 'smoke',
      paused: false
    };
    this.animParams = {
      colSphR: 0.9,
      colSphG: 0.5,
      colSphB: 0.6,
      colSkyR: 0.2,
      colSkyG: 0.7,
      colSkyB: 0.1,
      sphRad: 0.3,
      sphPos: 0.8,
      camFreq: 0.03,
      camHeight: 0.5
    };

    // Lights
    const ambient = new THREE.AmbientLight(0x999999);
    this.scene.add(ambient);

    const light = new THREE.PointLight(0xffffff, 1);
    light.position.set(0, 10, 0);
    if (this.vizParams.shadow) {
      light.castShadow = true;
    }
    this.scene.add(light);

    // Components
    this.comps = [
      new Spheres(this.vizParams, this.animParams),
      new Sky(this.vizParams, this.animParams)
    ];
    this.comps.forEach(comp => {
      this.scene.add(comp);
    });

    // Cache
    this.cache = {};
  }

  static getVisualParamsInfo() {
    return [
      ['colSphR', 'Sphere Colour Red'],
      ['colSphG', 'Sphere Colour Green'],
      ['colSphB', 'Sphere Colour Blue'],
      ['colSkyR', 'Background Colour Red'],
      ['colSkyG', 'Background Colour Green'],
      ['colSkyB', 'Background Colour Blue'],
      ['sphRad', 'Sphere Radius'],
      ['sphPos', 'Sphere Position (from centre)'],
      ['camFreq', 'Camera Revolution Speed'],
      ['camHeight', 'Camera Height']
    ]
  }

  animate(time) {
    // Camera revolution
    if (!this.cache.hasOwnProperty('time')) {
      this.cache.time = time;
      this.cache.camAngle = 0;
    }
    const interval = time - this.cache.time;
    const camAngle = (2 * Math.PI * this.animParams.camFreq * interval / this.vizParams.camFreqFactor) + this.cache.camAngle;
    this.cache.time = time;
    this.cache.camAngle = camAngle;
    this.camera.position.x = this.vizParams.camRad * Math.sin(camAngle);
    this.camera.position.z = this.vizParams.camRad * Math.cos(camAngle);

    this.camera.position.y = (2 * this.animParams.camHeight - 1) * this.vizParams.camHeightMax;
    this.camera.lookAt(0, 0, 0);

    this.comps.forEach(comp => {
      comp.animate && comp.animate(time);
    });
  }

  destroy() {
    this.comps.forEach(comp => {
      this.scene.remove(comp);
    });
    super.destroy();
  }
}
