import THREE from "../three";

import BaseViz from "./base-viz";
import Crystals from "./comps/crystals";
import Sky from "./comps/sky";

export default class VizCrystals extends BaseViz {
  constructor(container) {
    super(container);

    this.vizParams = {
      camRad: 500,
      skyRad: 700,
      numCrystalsPerGroup: 10,
      numCrystalGroups: 6,
      rotationSlowdown: 500,
      skyTexture: 'crystal',
      shadow: false, // true not supported
      paused: false
    };

    this.animParams = {
      rotXCrystal: 0.1,
      rotYCrystal: 0.1,
      rotZCrystal: 0.9,
      colSky: 0.2,
      sizeCrystal1: 0.2,
      sizeCrystal2: 0.2,
      sizeCrystal3: 0.2,
      sizeCrystal4: 0.2,
      sizeCrystal5: 0.2,
      sizeCrystal6: 0.2,
    };

    // Scene
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.offsetWidth / container.offsetHeight,
      0.1,
      10000
    );
    this.camera.position.z = this.vizParams.camRad;

    this.renderer = new THREE.WebGLRenderer({
      antialias: true
    });
    this.renderer.setSize(
      this.container.offsetWidth,
      this.container.offsetHeight
    );
    if (this.vizParams.shadow) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    this.container.appendChild(this.renderer.domElement);

    var ambient = new THREE.AmbientLight(0x999999);
    this.scene.add(ambient);

    // Components
    this.comps = [
      new Crystals(this.vizParams, this.animParams),
      new Sky(this.vizParams, this.animParams)
    ];
    this.comps.forEach(comp => {
      this.scene.add(comp);
    });

    this.initOrbitControls();

    // this.renderer.render(this.scene, this.camera);
    this.cache = {};
  }

  static getVisualParamsInfo() {
    return [
      ['rotXCrystal', 'Crystal Rotation Speed along X Axis'],
      ['rotYCrystal', 'Crystal Rotation Speed along Y Axis'],
      ['rotYCrystal', 'Crystal Rotation Speed along Z Axis'],
      ['colSky', 'Background Hue'],
      ['sizeCrystal1', 'Size of Crystal Group 1'],
      ['sizeCrystal2', 'Size of Crystal Group 2'],
      ['sizeCrystal3', 'Size of Crystal Group 3'],
      ['sizeCrystal4', 'Size of Crystal Group 4'],
      ['sizeCrystal5', 'Size of Crystal Group 5'],
      ['sizeCrystal6', 'Size of Crystal Group 6']
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
  }
}
