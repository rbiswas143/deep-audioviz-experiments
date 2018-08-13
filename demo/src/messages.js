const message_loading = '<span>' +
  '<i class="fa fa-2x fa-circle-o-notch fa-spin"></i>&nbsp;' +
  'Extracting features from the track...' +
  '</span>';

const message_error = 'Oops! Something went terribly wrong :,( ' +
  '<br> Check your internet connection, perhaps?';


export default class VizBoxMessages {

  constructor() {
    this.messageContainer = document.getElementById("message-box");
    this.init_message = this.messageContainer.innerHTML;
    this.cache = {};
  }

  reset() {
    this.messageContainer = document.getElementById("message-box");
    this.setInit();
  }

  updateMessage(show, message) {
    if (show && this.cache.hasOwnProperty('messageBoxDisplay')) {
      this.messageContainer.style.display = this.cache.messageBoxDisplay;
    } else if (!show) {
      this.cache.messageBoxDisplay = this.messageContainer.style.display;
      this.messageContainer.style.display = 'none';
    }
    if (message) {
      this.messageContainer.innerHTML = message;
    }
  }

  hide() {
    this.updateMessage(false);
  }

  setInit() {
    this.updateMessage(true, this.init_message);
  }

  setLoading() {
    this.updateMessage(true, message_loading);
  }

  setError() {
    this.updateMessage(true, message_error);
  }
}
