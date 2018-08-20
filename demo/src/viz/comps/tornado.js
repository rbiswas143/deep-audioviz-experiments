import THREE from "../../three";
import {scaleToRange} from "../utils";

export default class Tornado extends THREE.Group {
  constructor(vizParams, animParams) {
    super();
    this.vizParams = vizParams;
    this.animParams = animParams;

    // tornado lathe
    const points = [];
    for (let i = 0; i < 10; i++) {
      points.push(new THREE.Vector2(
        Math.sin(i * 0.2) * 8 + 1,
        i * 2
      ));
    }
    const latheGeo = new THREE.LatheBufferGeometry(points, 8);
    const latheMat = new THREE.MeshPhongMaterial({
      color: 0xd9edfd,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.3
    });
    this.lathe = new THREE.Mesh(latheGeo, latheMat);
    this.lathe.scale.setScalar(0.1);
    this.add(this.lathe);

    // torus ring column following lathe points
    this.ringGroup = new THREE.Group();
    this.add(this.ringGroup);
    const torusMat = new THREE.MeshPhongMaterial({
      color: 0xd9edfd,
      specular: 0x111111,
      shininess: 10
    });
    for (let i = 0; i < points.length; i++) {
      let radius = points[i].x * 0.1;
      let torusGeo = new THREE.TorusBufferGeometry(radius, i / points.length * 0.04 + 0.02, 12, 32);
      let torus = new THREE.Mesh(torusGeo, torusMat);
      torus.rotation.x = Math.PI / 2;
      torus.position.y = points[i].y * 0.1;
      this.ringGroup.add(torus);
    }

  }

  animate(timeStamp) {
    const latheRotation = scaleToRange(this.animParams.tornado, this.vizParams.tornadoLatheMin, this.vizParams.tornadoLatheMax);
    this.lathe.rotation.y -= latheRotation;

    // rotate rings
    const ringRotation = scaleToRange(this.animParams.tornado, this.vizParams.tornadoRingMin, this.vizParams.tornadoRingMax);
    let rings = this.ringGroup.children;
    for (let i = 0; i < rings.length; i++) {
      let ring = rings[i];
      ring.position.x = Math.cos(this.vizParams.time * i * ringRotation) * i / rings.length * 0.4;
      ring.position.z = Math.sin(this.vizParams.time * i * ringRotation) * i / rings.length * 0.4;
    }
  }

}
