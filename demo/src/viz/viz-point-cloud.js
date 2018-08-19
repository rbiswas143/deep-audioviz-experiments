import THREE from "../three";

import BaseViz from "./base-viz";

// Point textures
import flower1 from "../../assets/flower1_icon.png";
import flower2 from "../../assets/flower2_icon.png";
import flower3 from "../../assets/flower3_icon.png";
import flower4 from "../../assets/flower4_icon.png";
import flower5 from "../../assets/flower5_icon.png";
import flower6 from "../../assets/flower6_icon.png";
import leaf1 from "../../assets/leaf1_icon.png";
import leaf2 from "../../assets/leaf2_icon.png";
import leaf3 from "../../assets/leaf3_icon.png";

export default class VizPointCloud extends BaseViz {
  init() {
    this.vizParams = {
      camFov: 75,
      camNear: 0.1,
      camFar: 3000,
      camZ: 1000,
      orbitalControls: true,
      cloudSize: 500,
      numCloudGroups: 6,
      numCloudsPerGroup: 2,
      pointMaxSize: 18,
      cloudMaxRad: 1000,
      rotationSlowdown: 500,
      fogDensityLogRange: [-7, -2],
      paused: false
    };

    this.animParams = {
      hueCloud1: 0.9,
      hueCloud2: 0.5,
      hueCloud3: 0.6,
      hueCloud4: 0.2,
      rotSpeedCloud1: 0.7,
      rotSpeedCloud2: 0.1,
      rotSpeedCloud3: 0.3,
      rotSpeedCloud4: 0.8,
      fogDensity: 0.03,
      fogHue: 0.5
    };

    // Fog
    this.scene.fog = new THREE.FogExp2(0x000000, 0);

    // Point Clouds
    this.pointClouds = [];
    this.initPointCloud();
  }

  initPointCloud() {

    // Common cloud geometry
    const geometry = new THREE.Geometry();
    for (let i = 0; i < this.vizParams.cloudSize; i++) {
      let vertex = new THREE.Vector3();
      vertex.x = (2 * Math.random() - 1) * this.vizParams.cloudMaxRad;
      vertex.y = (2 * Math.random() - 1) * this.vizParams.cloudMaxRad;
      vertex.z = (2 * Math.random() - 1) * this.vizParams.cloudMaxRad;
      geometry.vertices.push(vertex);
    }

    // All textures
    const sprites = [flower1, flower2, flower3, flower4,
      flower5, flower6, leaf1, leaf2, leaf3];

    // Sprite rotation angles
    const rotAngles = [0, 0.25, 0.5, 0.75].map(n => 2 * Math.PI * n);

    for (let i = 0; i < this.vizParams.numCloudGroups; i++) {
      this.pointClouds.push([]);

      for (let j = 0; j < this.vizParams.numCloudsPerGroup; j++) {

        // Material with random size
        let spriteIndex = Math.min(Math.floor(Math.random() * sprites.length), sprites.length - 1);
        let sprite = new THREE.TextureLoader().load(sprites[spriteIndex]);
        sprite.rotation = rotAngles[Math.min(Math.floor(Math.random() * rotAngles.length), rotAngles.length - 1)];
        sprite.center = new THREE.Vector2(0.5, 0.5);
        let material = new THREE.PointsMaterial({
          size: Math.min(Math.floor(Math.random() * (this.vizParams.pointMaxSize + 1)), this.vizParams.pointMaxSize),
          map: sprite,
          blending: THREE.AdditiveBlending,
          depthTest: false,
          transparent: false
        });

        // New cloud with random orientation
        let cloud = new THREE.Points(geometry, material);
        cloud.rotation.x = Math.random() * 6;
        cloud.rotation.y = Math.random() * 6;
        cloud.rotation.z = Math.random() * 6;

        this.scene.add(cloud);
        this.pointClouds[i][j] = cloud;
      }
    }
  }

  static getVisualParamsInfo() {
    return [
      ['hueCloud1', 'Hue of Point Cloud 1'],
      ['hueCloud2', 'Hue of Point Cloud 2'],
      ['hueCloud3', 'Hue of Point Cloud 3'],
      ['hueCloud4', 'Hue of Point Cloud 4'],
      ['rotSpeedCloud1', 'Rotation Speed of Point Cloud 1'],
      ['rotSpeedCloud2', 'Rotation Speed of Point Cloud 2'],
      ['rotSpeedCloud3', 'Rotation Speed of Point Cloud 3'],
      ['rotSpeedCloud4', 'Rotation Speed of Point Cloud 4'],
      ['fogDensity', 'Fog Density'],
      ['fogHue', 'Fog Hue']
    ]
  }

  animate(time) {

    const rotationFactor = 2 * Math.PI / this.vizParams.rotationSlowdown;

    // Update fog hue
    let fogHue = Math.floor(this.animParams[`fogHue`] * 360);
    let fogColor = new THREE.Color(`hsl(${fogHue}, 15%, 20%)`);
    this.scene.fog.color.set(fogColor);
    this.renderer.setClearColor(fogColor); // Background color is same as fog color

    // Update fog density
    const [densityRangeStart, densityRangeEnd] = this.vizParams.fogDensityLogRange;
    const densityExp = densityRangeStart + (this.animParams.fogDensity * (densityRangeEnd - densityRangeStart));
    this.scene.fog.density = Math.pow(10, densityExp);

    // Update point clouds
    for (let i = 0; i < this.vizParams.numCloudGroups; i++) {
      let group_index = i + 1;

      // Cloud Group Hue
      let hue = Math.floor(this.animParams[`hueCloud${group_index}`] * 360);
      let color = new THREE.Color(`hsl(${hue}, 80%, 70%)`);

      // Cloud Group Rotation
      let rotationChange = (this.animParams[`rotSpeedCloud${group_index}`] - 0.5) * rotationFactor;

      for (let j = 0; j < this.vizParams.numCloudsPerGroup; j++) {
        let cloud = this.pointClouds[i][j];
        cloud.material.color.set(color);
        cloud.rotation.y += rotationChange;
      }
    }

  }
}
