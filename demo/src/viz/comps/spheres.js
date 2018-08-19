import THREE from "../../three";

// Sphere texture
import sphereTexture from "../../../assets/disco-ball.jpeg";

export default class Spheres extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    this.sphereUnitVectors = [];
    this.spheres = [];

    this.texture = new THREE.TextureLoader().load(sphereTexture);
    this.texture.repeat.set(0.7, 1);
    this.texture.wrapS = this.texture.wrapT = THREE.RepeatWrapping;

    // Create spheres
    for (let i = -1; i < 2; i++) {
      for (let j = -1; j < 2; j++) {
        for (let k = -1; k < 2; k++) {
          if (i === 0 && j === 0 && k === 0) {
            continue;
          }

          // Unit vector
          const magnitude = Math.sqrt(
            Math.pow(i, 2) + Math.pow(j, 2) + Math.pow(k, 2)
          );
          const unitVector = new THREE.Vector3(
            i / magnitude,
            j / magnitude,
            k / magnitude
          );
          this.sphereUnitVectors.push(unitVector);

          // Geometry
          const geometry = new THREE.SphereGeometry(
            this.animParams.sphRad,
            32,
            32
          );

          // Material
          const material = new THREE.MeshPhongMaterial({
            // ambient: 0x444444,
            color: 0x8844aa,
            shininess: 3,
            specular: 0xaaaaff,
            // shading: THREE.SmoothShading,
            map: this.texture
          });

          // Mesh
          const sphere = new THREE.Mesh(geometry, material);

          // Position
          const scaledVector = unitVector
            .clone()
            .multiplyScalar(this.animParams.sphPos);
          sphere.position.set(
            scaledVector.x,
            scaledVector.y,
            scaledVector.z
          );

          // Shadow
          if (this.vizParams.shadow) {
            sphere.castShadow = true;
            sphere.receiveShadow = false;
          }

          this.add(sphere);
          this.spheres.push(sphere);
        }
      }
    }
  }

  animate(timeStamp) {
    for (let i = 0; i < this.spheres.length; i++) {
      const sphere = this.spheres[i];

      // Update size (Radius)
      const geometry = new THREE.SphereGeometry(
        this.animParams.sphRad * this.vizParams.sphRadMax,
        32,
        32
      );
      sphere.geometry.dispose();
      sphere.geometry = geometry;
      geometry.verticesNeedUpdate = true;

      // Update color
      sphere.material.color.set(
        new THREE.Color(
          this.animParams.colSphR,
          this.animParams.colSphG,
          this.animParams.colSphB
        )
      );

      // Update position (distance from origin)
      const unitVector = this.sphereUnitVectors[i];
      const sphPos = this.vizParams.sphPosMin + (this.animParams.sphPos *
        (this.vizParams.sphPosMax - this.vizParams.sphPosMin));
      const scaledUnitVector = unitVector
        .clone()
        .multiplyScalar(sphPos);
      sphere.position.set(
        scaledUnitVector.x,
        scaledUnitVector.y,
        scaledUnitVector.z
      );
    }
  }
}
