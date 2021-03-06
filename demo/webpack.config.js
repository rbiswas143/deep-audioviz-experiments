const webpack = require("webpack");
const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const buildPath = "./build/";
const Dotenv = require('dotenv-webpack');

module.exports = (env, argv) => {
  console.log('Environment:', argv.mode);

  return {
    entry: ["./src/main.js"],
    output: {
      path: path.join(__dirname, buildPath),
      filename: "[name].[hash].js"
    },
    target: "web",
    devtool: "source-map",
    module: {
      rules: [
        {
          test: /\.js$/,
          use: "babel-loader",
          exclude: path.resolve(__dirname, "./node_modules/")
        },
        {
          test: /\.(jpe?g|png|gif|svg|tga|gltf|babylon|mtl|pcb|pcd|prwm|obj|mat|mp3|ogg)$/i,
          use: "file-loader",
          exclude: path.resolve(__dirname, "./node_modules/")
        },
        {
          test: /\.(vert|frag|glsl|shader|txt)$/i,
          use: "raw-loader",
          exclude: path.resolve(__dirname, "./node_modules/")
        },
        {
          type: "javascript/auto",
          test: /\.(json)/,
          exclude: path.resolve(__dirname, "./node_modules/"),
          use: [
            {
              loader: "file-loader"
            }
          ]
        },
        {
          test: /\.css$/,
          use: [
            {loader: "style-loader"},
            {loader: "css-loader"}
          ]
        }
      ]
    },
    plugins: [
      new HtmlWebpackPlugin({
        hash: true,
        title: "Deep AudioViz Experiments",
        template: "./src/index.html",
        filename: path.join(__dirname, buildPath, "index.html"),
	favicon: "assets/favicon.ico"
      }),
      new Dotenv({
        path: `./.env.${argv.mode === "production" ? "prod" : "dev"}`,
      })
    ]
  }
};
