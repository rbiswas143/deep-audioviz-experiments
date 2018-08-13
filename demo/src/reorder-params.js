import $ from 'jquery';

require('../node_modules/jquery-ui/ui/widgets/sortable');

import {vizMap} from "./viz/all";

export default class VizParamsReordering {

  constructor() {
    this.$vizSelect = $("#viz-select");
    this.$vizParamsList = $('#viz-params-list').sortable();
    this.$vizNameSpan = $('.curr-viz-name');
    this.$vizSelect.change(() => this.updateVizParams());
    this.updateVizParams();
  }

  updateVizParams() {
    const vizName = this.$vizSelect.find('option:selected').text();
    this.$vizNameSpan.text(vizName);
    const vizCode = this.$vizSelect.val();
    const paramsInfo = vizMap[vizCode].getVisualParamsInfo();
    const paramsList = document.querySelectorAll('.viz-param-item');
    for (let i = 0; i < paramsList.length; i++) {
      paramsList[i].setAttribute('viz-param-name', paramsInfo[i][0]);
      paramsList[i].innerText = paramsInfo[i][1];
    }
  }
}
