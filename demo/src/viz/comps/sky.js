import THREE from "../../three";
import smokeTexture from "../../../assets/smoke.jpg";
import crystalTexture from "../../../assets/crystal.jpeg";

export default class Sky extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    const geometry = new THREE.SphereGeometry(this.vizParams.skyRad, 32, 32);
    let texture = null;
    if (this.vizParams.skyTexture === 'smoke') {
      texture = new THREE.TextureLoader().load(smokeTexture);
    } else if (this.vizParams.skyTexture === 'crystal') {
      texture = new THREE.TextureLoader().load(crystalTexture);
    }
    const material = new THREE.MeshPhongMaterial({
      // ambient: 0x444444,
      color: 0x66aa66,
      shininess: 150,
      specular: 0x888888,
      // shading: THREE.SmoothShading,
      map: texture
    });
    material.side = THREE.DoubleSide;
    this.sky = new THREE.Mesh(geometry, material);
    if (this.vizParams.shadow) {
      this.sky.castShadow = false;
      this.sky.receiveShadow = true;
    }
    this.add(this.sky);
  }

  animate(timeStamp) {
    let color = null;
    if (this.animParams.hasOwnProperty('colSky')) {
      const hue = Math.floor(this.animParams.colSky * 360);
      color = new THREE.Color(`hsl(${hue}, 70%, 50%)`);
    } else {
      color = new THREE.Color(
        this.animParams.colSkyR,
        this.animParams.colSkyG,
        this.animParams.colSkyB
      );
    }
    this.sky.material.color.set(color);
  }
}
