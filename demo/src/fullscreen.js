// Adapted from "https://github.com/typpo/millenium-viz/blob/master/web/vendor/threex/THREEx.FullScreen.js" to be used without THREEx

// internal functions to know which fullscreen API implementation is available
const _hasWebkitFullScreen = 'webkitCancelFullScreen' in document ? true : false;
const _hasMozFullScreen = 'mozCancelFullScreen' in document ? true : false;

export const available = function () {
  return _hasWebkitFullScreen || _hasMozFullScreen;
};

export const activated = function () {
  if (_hasWebkitFullScreen) {
    return document.webkitIsFullScreen;
  } else if (_hasMozFullScreen) {
    return document.mozFullScreen;
  } else {
    console.assert(false);
  }
};

export const request = function (element) {
  element = element || document.body;
  if (_hasWebkitFullScreen) {
    element.webkitRequestFullScreen(Element.ALLOW_KEYBOARD_INPUT);
  } else if (_hasMozFullScreen) {
    element.mozRequestFullScreen();
  } else {
    console.assert(false);
  }
};

export const cancel = function () {
  if (_hasWebkitFullScreen) {
    document.webkitCancelFullScreen();
  } else if (_hasMozFullScreen) {
    document.mozCancelFullScreen();
  } else {
    console.assert(false);
  }
};

export const changeSuccessCallback = function (callback) {
  ['webkitfullscreenchange', 'mozfullscreenchange', 'fullscreenchange'].forEach(e => {
    document.addEventListener(e, callback)
  });
};

export const toggle = function (element) {
  if (activated()) {
    cancel();
  } else {
    request(element);
  }
};

export const bindKey = function (opts) {
  opts = opts || {};
  var charCode = opts.charCode || 'f'.charCodeAt(0);
  var dblclick = opts.dblclick !== undefined ? opts.dblclick : false;
  var element = opts.element;

  // callback to handle keypress
  var __bind = function (fn, me) {
    return function () {
      return fn.apply(me, arguments);
    };
  };
  var onKeyPress = __bind(function (event) {
    // return now if the KeyPress isnt for the proper charCode
    if (event.which !== charCode) return;
    // toggle fullscreen
    toggle(element);
  }, this);

  // listen to keypress
  // NOTE: for firefox it seems mandatory to listen to document directly
  document.addEventListener('keypress', onKeyPress, false);
  // listen to dblclick
  dblclick && (element || document).addEventListener('dblclick', () => toggle(element), false);

  return {
    unbind: function () {
      document.removeEventListener('keypress', onKeyPress, false);
      dblclick && (element || document).removeEventListener('dblclick', () => toggle(element), false);
    }
  };
};
