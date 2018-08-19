import THREE from "../../three";

// Crystal Shape
const hexShape = new THREE.Shape();
hexShape.moveTo(0, 0.8);
hexShape.lineTo(0.4, 0.5);
hexShape.lineTo(0.3, 0);
hexShape.lineTo(-0.3, 0);
hexShape.lineTo(-0.4, 0.5);
hexShape.lineTo(0, 0.8);

export default class Crystals extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    // Create crystals
    this.crystals = [];
    this.numCrystals = this.vizParams.numCrystalGroups * this.vizParams.numCrystalsPerGroup;
    for (let i = 0; i < this.numCrystals; i++) {
      this.addShape(
        hexShape,
        0xff3333, // color
        0, // x pos
        0, // y pos
        0, // z pos
        Math.random() * 2 * Math.PI, // x rotation
        Math.random() * 2 * Math.PI, // y rotation
        Math.random() * 2 * Math.PI, // z rotation
        1
      );
    }

    // Crystal Groups
    this.crystalGroups = [];
    for (let i = 0; i < this.numCrystals; i++) {
      this.crystalGroups.push(i);
    }
    this.crystalGroups.sort(() => 0.5 - Math.random());

  }

  getExtrudeGeometry(depth, bevelSize, bevelThickness) {
    const extrudeSettings = {
      depth: depth, // Orig: 0 to 100
      bevelEnabled: true,
      bevelSegments: 1,
      steps: 1,
      bevelSize: bevelSize, // Orig: 15 to 25
      bevelThickness: bevelThickness // Orig 25 to 35
    };
    return new THREE.ExtrudeGeometry(hexShape, extrudeSettings);
  }

  addShape(shape, color, x, y, z, rx, ry, rz, s) {
    const geometry = this.getExtrudeGeometry(50, 20, 30);
    const meshMaterial = new THREE.MeshNormalMaterial();
    const mesh = new THREE.Mesh(geometry, meshMaterial);
    mesh.position.set(x, y, z);
    mesh.rotation.set(rx, ry, rz);
    mesh.scale.set(s, s, s);
    if (this.vizParams.shadow) {
      mesh.castShadow = true;
      mesh.receiveShadow = false;
    }
    this.crystals.push(mesh);
    this.add(mesh);
  }

  animate(timeStamp) {

    // Group Rotation
    const rot_factor = 2 * Math.PI / this.vizParams.rotationSlowdown;
    this.rotation.x += rot_factor * this.animParams.rotXCrystal;
    this.rotation.y += rot_factor * this.animParams.rotYCrystal;
    this.rotation.z += rot_factor * this.animParams.rotZCrystal;

    // Crystal animations
    for (let i = 0; i < this.vizParams.numCrystalGroups; i++) {
      let group_index = i + 1;
      for (let j = 0; j < this.vizParams.numCrystalsPerGroup; j++) {
        let crystal_index = i * this.vizParams.numCrystalsPerGroup + j;
        let crystal = this.crystals[crystal_index];
        crystal.geometry.dispose();
        let crystalDepth = this.animParams[`sizeCrystal${group_index}`] * 300;
        let bevelSize = 15 + (this.animParams[`sizeCrystal${group_index}`] * 10);
        let bevelThickness = 25 + (this.animParams[`sizeCrystal${group_index}`] * 30);
        let geometry = this.getExtrudeGeometry(crystalDepth, bevelSize, bevelThickness);
        crystal.geometry = geometry;
        geometry.verticesNeedUpdate = true;
      }
    }
  }

}
