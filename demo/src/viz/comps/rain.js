import THREE from "../../three";
import {scaleToRange} from "../utils";

export default class Rain extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    // Position
    this.position.y = 2;

    // Rain texture
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = canvas.width / 3;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
    ctx.fillStyle = '#fff';
    ctx.fill();
    const texture = new THREE.Texture(canvas);
    texture.premultiplyAlpha = true;
    texture.needsUpdate = true;

    // Material
    const pointsMat = new THREE.PointsMaterial({
      color: 0x2e2833,
      size: 0.1,
      map: texture,
      transparent: true,
      depthWrite: false
    });

    // Geometry
    const pointsGeo = new THREE.Geometry();
    this.pointCount = this.vizParams.numRainDrops;
    this.rangeV = 4; // 600
    this.rangeH = 8;
    for (let p = 0; p < this.pointCount; p++) {
      let point = new THREE.Vector3(
        THREE.Math.randFloatSpread(this.rangeH),
        THREE.Math.randFloatSpread(this.rangeV),
        THREE.Math.randFloatSpread(this.rangeH)
      );
      point.velocity = new THREE.Vector3(0, -Math.random() * 0.05, 0);
      pointsGeo.vertices.push(point);
    }

    // Create Points
    this.points = new THREE.Points(pointsGeo, pointsMat);
    this.points.position.y = -this.rangeV / 2;
    this.points.sortParticles = true;
    this.add(this.points);

  }

  animate(timeStamp) {
    this.points.rotation.y -= 0.003;
    const rainVelocityLog = scaleToRange(this.animParams.rain, this.vizParams.rainMin, this.vizParams.rainMax);
    const rainVelocity = Math.pow(10, rainVelocityLog);

    let pCount = this.pointCount;
    while (pCount--) {

      let point = this.points.geometry.vertices[pCount];

      // check if we need to reset
      if (point.y < -this.rangeV / 2) {
        point.y = this.rangeV / 2;
        // point.velocity.y = 0;
      }

      // update the velocity
      point.velocity.y -= Math.random() * rainVelocity;

      // and the position
      point.add(point.velocity);
    }

    this.points.geometry.verticesNeedUpdate = true;
  }

}
