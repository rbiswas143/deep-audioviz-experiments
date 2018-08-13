import THREE from "../../three";
import planetTexture from "../../../assets/disco-ball.jpeg";

export default class Planets extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    this.planetUnitVectors = [];
    this.planets = [];

    this.texture = new THREE.TextureLoader().load(planetTexture);
    this.texture.repeat.set(0.7, 1);
    this.texture.wrapS = this.texture.wrapT = THREE.RepeatWrapping;

    for (let i = -1; i < 2; i++) {
      for (let j = -1; j < 2; j++) {
        for (let k = -1; k < 2; k++) {
          if (i === 0 && j === 0 && k === 0) {
            continue;
          }
          const magnitude = Math.sqrt(
            Math.pow(i, 2) + Math.pow(j, 2) + Math.pow(k, 2)
          );
          const unitVector = new THREE.Vector3(
            i / magnitude,
            j / magnitude,
            k / magnitude
          );
          this.planetUnitVectors.push(unitVector);
          const geometry = new THREE.SphereGeometry(
            this.animParams.sphRad,
            32,
            32
          );
          const material = new THREE.MeshPhongMaterial({
            // ambient: 0x444444,
            color: 0x8844aa,
            shininess: 300,
            specular: 0x33aa33,
            // shading: THREE.SmoothShading,
            map: this.texture
          });

          const planet = new THREE.Mesh(geometry, material);
          const scaledVector = unitVector
            .clone()
            .multiplyScalar(this.animParams.sphPos);
          planet.position.set(
            scaledVector.x,
            scaledVector.y,
            scaledVector.z
          );
          if (this.vizParams.shadow) {
            planet.castShadow = true;
            planet.receiveShadow = false;
          }
          this.add(planet);
          this.planets.push(planet);
        }
      }
    }
  }

  animate(timeStamp) {
    for (let i = 0; i < this.planets.length; i++) {
      const planet = this.planets[i];

      const geometry = new THREE.SphereGeometry(
        this.animParams.sphRad * this.vizParams.sphRadMax,
        32,
        32
      );
      planet.geometry.dispose();
      planet.geometry = geometry;
      geometry.verticesNeedUpdate = true;

      // Update color
      planet.material.color.set(
        new THREE.Color(
          this.animParams.colSphR,
          this.animParams.colSphG,
          this.animParams.colSphB
        )
      );

      // Update position
      const unitVector = this.planetUnitVectors[i];
      const sphPos = this.vizParams.sphPosMin + (this.animParams.sphPos *
        (this.vizParams.sphPosMax - this.vizParams.sphPosMin));
      const scaledUnitVector = unitVector
        .clone()
        .multiplyScalar(sphPos);
      planet.position.set(
        scaledUnitVector.x,
        scaledUnitVector.y,
        scaledUnitVector.z
      );
    }
  }
}
