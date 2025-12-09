const {
  registerUser,
  loginUser,
  logOut,
  demoLogin,
  verifyOTP,
  resendOTP,
  refreshAccessToken
} = require("../../controllers/auth");
const express = require("express");
const Router = express.Router();

Router.post("/register", registerUser);
Router.post("/verify-otp", verifyOTP);
Router.post("/resend-otp", resendOTP);
Router.post("/login", loginUser);
Router.post("/refresh", refreshAccessToken);
Router.post("/logout", logOut);
Router.post("/demo-login", demoLogin);

module.exports = Router;