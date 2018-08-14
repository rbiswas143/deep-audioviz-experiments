import THREE from "../three";

export default class BaseViz {
  constructor(container) {
    this.container = container;
    this.vizParams = null;
    this.animParams = null;
    this.orbitalControls = true;

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      75,
      container.offsetWidth / container.offsetHeight,
      0.1,
      3000
    );

    // Renderer
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // Viz specific init
    this.init();

    // Update camera
    this.vizParams.hasOwnProperty('camFov') && (this.camera.fov = this.vizParams.camFov);
    this.vizParams.hasOwnProperty('camNear') && (this.camera.near = this.vizParams.camNear);
    this.vizParams.hasOwnProperty('camFar') && (this.camera.far = this.vizParams.camFar);
    this.vizParams.hasOwnProperty('camX') && (this.camera.position.x = this.vizParams.camX);
    this.vizParams.hasOwnProperty('camY') && (this.camera.position.y = this.vizParams.camY);
    this.vizParams.hasOwnProperty('camZ') && (this.camera.position.z = this.vizParams.camZ);

    // Shadow
    if (this.vizParams.shadow) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    this.container.appendChild(this.renderer.domElement);
    this.initOrbitControls()
  }

  init() {
    throw 'Not Implemented';
  }

  static getVisualParamsInfo() {
    throw 'Not Implemented';
  }

  initOrbitControls() {
    if (this.vizParams.orbitalControls) {
      var controls = new THREE.OrbitControls(
        this.camera,
        this.renderer.domElement
      );
      controls.addEventListener("change", () => {
        this.renderer.render(this.scene, this.camera);
      });
    }
  }

  animate() {
    throw 'Not Implemented';
  }

  start() {
    requestAnimationFrame(() => this.start());
    this.resize();
    if (this.vizParams.paused) return;

    var time = new Date().getTime() / 1000;
    this.animate(time);

    this.renderer.render(this.scene, this.camera);
  }

  destroy() {
    this.container.innerHTML = '';
  }

  resize(force) {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (force || canvas.width !== width || canvas.height !== height) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
  }

}
