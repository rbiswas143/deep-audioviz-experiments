import THREE from "../three";

export default class BaseViz {
  constructor(container) {
    this.container = container;
    this.vizParams = null;
    this.animParams = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;

    window.addEventListener('resize', () => this.resize(), false);
  }

  static getVisualParamsInfo() {
    throw 'Not Implemented';
  }

  initOrbitControls() {
    var controls = new THREE.OrbitControls(
      this.camera,
      this.renderer.domElement
    );
    controls.addEventListener("change", () => {
      this.renderer.render(this.scene, this.camera);
    });
  }

  animate() {
    throw 'Not Implemented';
  }

  start() {
    requestAnimationFrame(() => this.start());
    if (this.vizParams.paused) return;

    var time = new Date().getTime() / 1000;
    this.animate(time);

    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
  }

  resize() {
    this.camera.aspect = this.container.offsetWidth / this.container.offsetHeight;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(
      this.container.offsetWidth,
      this.container.offsetHeight
    );

  }
}
