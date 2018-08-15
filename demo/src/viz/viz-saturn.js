import THREE from "../three";

import BaseViz from "./base-viz";
import Saturn from "./comps/saturn";
import Sky from "./comps/sky";

export default class VizSaturn extends BaseViz {
  init() {

    this.vizParams = {
      camFov: 75,
      camNear: 0.1,
      camFar: 2000,
      camX: 600,
      camY: 0,
      camZ: 0,
      orbitalControls: true,
      planetShapeDetail: 2,
      planetMinRad: 120,
      planetMaxRad: 200,
      planetMorphMax: 60,
      numDiscs: 6,
      numParticlesPerGroup: 15,
      rotationSlowdown: 300,
      discParticleMaxDetail: 2,
      discParticleMinRad: 10,
      discParticleMaxRad: 30,
      discInnerRadMin: 200,
      discInnerRadMax: 300,
      discOuterRadMin: 300,
      discOuterRadMax: 500,
      discParticleMinOscillation: 10,
      discParticleMaxOscillation: 100,
      maxAxisTilt: 1,
      skyRad: 800,
      skyShininess: 0,
      skyTexture: 'galaxy',
      paused: false
    };

    this.animParams = {
      planetHue: 0.1,
      planetMorph: 0.1,
      planetSize: 0.9,
      discRotation: 0.2,
      discHue: 0.2,
      discParticlesSize: 0.2,
      discInnerRad: 0.2,
      discOuterRad: 0.2,
      discOscillation: 0.2,
      axisTilt: 0.2,
    };

    // Renderer
    this.renderer.alpha = true;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x663344, 2);
    this.scene.add(ambientLight);

    const light = new THREE.DirectionalLight(0xffffff, 1.5);
    light.position.set(200, 100, 200);
    // light.castShadow = true;
    light.shadow.camera.left = -400;
    light.shadow.camera.right = 400;
    light.shadow.camera.top = 400;
    light.shadow.camera.bottom = -400;
    light.shadow.camera.near = 1;
    light.shadow.camera.far = 1000;
    light.shadow.mapSize.width = 2048;
    light.shadow.mapSize.height = 2048;
    this.scene.add(light);

    // Components
    this.comps = [
      new Saturn(this.vizParams, this.animParams),
      new Sky(this.vizParams, this.animParams)
    ];
    this.comps.forEach(comp => {
      this.scene.add(comp);
    });
  }

  static getVisualParamsInfo() {
    return [
      ['planetHue', 'Planet Hue'],
      ['planetMorph', 'Planet Morphing'],
      ['planetSize', 'Planet Size'],
      ['discRotation', 'Rotation Speed of Disc'],
      ['discHue', 'Disc Hue'],
      ['discParticlesSize', 'Size of the Disc Particles'],
      ['discInnerRad', 'Inner Radius of the Disc'],
      ['discOuterRad', 'Outer Radius od the Disc'],
      ['discOscillation', 'Oscillations of the Disc Particles'],
      ['axisTilt', "Tilt of the Planet's Axis"]
    ]
  }

  animate(time) {
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
