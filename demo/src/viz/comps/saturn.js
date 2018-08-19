import THREE from "../../three";

// Texture for planet
import saturnTexture from "../../../assets/saturn.jpeg";

// Textures for disc particles
import moonTexture1 from "../../../assets/moon1.png";
import moonTexture2 from "../../../assets/moon2.png";
import moonTexture3 from "../../../assets/moon3.png";
import moonTexture4 from "../../../assets/moon4.png";


export default class Saturn extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    // Planet
    this.createPlanet();

    // Disc
    this.createDisc();
  }

  createPlanet() {
    // Geometry
    const geometry = new THREE.TetrahedronGeometry(20, this.vizParams.planetShapeDetail);

    // Material
    const texture = new THREE.TextureLoader().load(saturnTexture);
    texture.repeat.set(0.7, 1);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    const material = new THREE.MeshPhongMaterial({
      // ambient: 0x444444,
      color: 0x8844aa,
      shininess: 5,
      specular: 0x33aa33,
      // shading: THREE.SmoothShading,
      map: texture
    });

    // Mesh
    this.planet = new THREE.Mesh(geometry, material);
    this.add(this.planet);

    // Save Unit vectors
    this.planetVerticesUnitVectors = geometry.vertices.map(vertex => {
      const magnitude = Math.sqrt(Math.pow(vertex.x, 2) + Math.pow(vertex.y, 2) + Math.pow(vertex.z, 2));
      return new THREE.Vector3(
        vertex.x / magnitude,
        vertex.y / magnitude,
        vertex.z / magnitude
      );
    });

    // Generate random phases for oscillations of planet's vertices
    this.planetVerticesPhases = geometry.vertices.map(() => Math.random() * 6);
  }

  createDisc() {
    const textures = [moonTexture1, moonTexture2,
      moonTexture3, moonTexture4].map(t => new THREE.TextureLoader().load(t));

    this.discGroups = [];
    this.particlePositions = [];
    this.particleDetals = [];
    this.rotationFactors = [];
    this.sizeFactors = [];

    for (let i = 0; i < this.vizParams.numDiscs; i++) {

      // Each disc is a group
      let currGroup = this.discGroups[i] = new THREE.Group();
      this.add(currGroup);

      // Speed and size factors for current group (so all discs do not appear the same)
      this.rotationFactors[i] = 0.5 + (Math.random() * 1.5);
      this.sizeFactors[i] = 0.5 + (Math.random() * 1.5);

      // Positions
      let currPositions = this.particlePositions[i] = [];
      let currDetails = this.particleDetals[i] = [];

      // Create
      for (let j = 0; j < this.vizParams.numParticlesPerGroup; j++) {
        // Particle Geometry with random detail
        let detail = Math.floor(Math.random() * (this.vizParams.discParticleMaxDetail + 1));
        currDetails.push(detail);
        let geometry = new THREE.OctahedronGeometry(5, detail);

        // Particle material with random texture
        let textureIndex = Math.min(textures.length, Math.floor(Math.random() * textures.length + 1));
        let material = new THREE.MeshPhongMaterial({
          // ambient: 0x444444,
          color: 0x8844aa,
          shininess: 5,
          specular: 0x33aa33,
          // shading: THREE.SmoothShading,
          map: textures[textureIndex]
        });

        // Mesh
        let particle = new THREE.Mesh(geometry, material);

        // Random position
        let [xPos, yPos, zPos] = [2 * Math.random() - 1, 0, 2 * Math.random() - 1];
        particle.position.set(xPos, yPos, zPos);

        // Save Unit Vectors and Magnitude
        let magnitude = Math.sqrt(Math.pow(xPos, 2) + Math.pow(yPos, 2) + Math.pow(zPos, 2));
        let uv = new THREE.Vector3(xPos / magnitude, yPos / magnitude, zPos / magnitude);
        currPositions.push([uv, magnitude]);

        currGroup.add(particle);
      }
    }
  }

  animate(timeStamp) {

    // Planet Color
    const planetHue = Math.floor(this.animParams.planetHue * 360);
    const color = new THREE.Color(`hsl(${planetHue}, 70%, 50%)`);
    this.planet.material.color.set(color);

    // Planet Size
    const newPlanetSize = this.vizParams.planetMinRad + ((this.vizParams.planetMaxRad -
      this.vizParams.planetMinRad) * this.animParams.planetSize);

    // Planet Morphing (vertex position is calculated using new size and morphing)
    for (let i = 0; i < this.planet.geometry.vertices.length; i++) {
      let planetVertexUnitVector = this.planetVerticesUnitVectors[i];
      let planetVertexPhase = this.planetVerticesPhases[i];
      let vertexLength = newPlanetSize + ((2 * this.animParams.planetMorph - 1) *
        this.vizParams.planetMorphMax * Math.sin(planetVertexPhase + timeStamp));
      this.planet.geometry.vertices[i].x = planetVertexUnitVector.x * vertexLength;
      this.planet.geometry.vertices[i].y = planetVertexUnitVector.y * vertexLength;
      this.planet.geometry.vertices[i].z = planetVertexUnitVector.z * vertexLength;
    }
    this.planet.geometry.verticesNeedUpdate = true;

    // Slow down overall disc rotation by a factor
    const allDiscsRotFactor = 2 * Math.PI / this.vizParams.rotationSlowdown;

    // Inner and Outer radius of all discs
    const discInnerRad = this.vizParams.discInnerRadMin + ((this.vizParams.discInnerRadMax -
      this.vizParams.discInnerRadMin) * this.animParams.discInnerRad);
    const discOuterRad = this.vizParams.discOuterRadMin + ((this.vizParams.discOuterRadMax -
      this.vizParams.discOuterRadMin) * this.animParams.discOuterRad);

    for (let i = 0; i < this.vizParams.numDiscs; i++) {
      let currGroup = this.discGroups[i];

      // Rotate disc
      currGroup.rotation.y += this.rotationFactors[i] * allDiscsRotFactor * this.animParams.discRotation;

      for (let j = 0; j < currGroup.children.length; j++) {
        let particle = currGroup.children[j];

        // Hue
        const particleHue = Math.floor(this.animParams.discHue * 360);
        const color = new THREE.Color(`hsl(${particleHue}, 70%, 50%)`);
        particle.material.color.set(color);

        // Size
        let particleSize = this.vizParams.discParticleMinRad + ((this.vizParams.discParticleMaxRad -
          this.vizParams.discParticleMinRad) * this.animParams.discParticlesSize * this.sizeFactors[i]);
        let newGeometry = new THREE.OctahedronGeometry(particleSize, this.particleDetals[i][j]);
        particle.geometry.dispose();
        particle.geometry = newGeometry;
        particle.geometry.verticesNeedUpdate = true;

        // Disc Size
        let [particleUnitVector, magnitude] = this.particlePositions[i][j];
        let particleScaleBy = discInnerRad + (magnitude * Math.max(0, discOuterRad - discInnerRad));
        particle.position.x = particleUnitVector.x * particleScaleBy;
        particle.position.z = particleUnitVector.z * particleScaleBy;

        // Oscillations
        let oscillationAmplitude = this.vizParams.discParticleMinOscillation + ((this.vizParams.discParticleMaxOscillation
          - this.vizParams.discParticleMinOscillation) * this.animParams.discOscillation);
        particle.position.y = oscillationAmplitude * Math.sin(2 * Math.PI *
          currGroup.rotation.y / this.vizParams.discParticleOscillationFreq);
      }

    }

    // Tilt
    this.rotation.x = this.vizParams.maxAxisTilt * this.animParams.axisTilt;
  }
}
