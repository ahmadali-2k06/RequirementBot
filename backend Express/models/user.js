const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const UserSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, "Please provide name"],
    minLength: 3,
    maxlength: 50,
  },
  email: {
    type: String,
    required: [true, "Please provide E-mail"],
    match: [
      /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/,
      "Please provide a valid E-mail",
    ],
    unique: true,
  },
  password: {
    type: String,
    required: [true, "Please provide password"],
    minLength: 8,
  },
  organization: {
    type: String,
    maxlength: 100,
    default: null,
  },
  location: {
    type: String,
    default: "My city",
  },
  refreshTokens: {
    type: [String],
    default: [],
  },
  // OTP Fields
  otp: {
    type: String,
    default: null,
  },
  otpExpiry: {
    type: Date,
    default: null,
  },
  isVerified: {
    type: Boolean,
    default: false,
  },
});

UserSchema.pre("save", async function (next) {
  if (!this.isModified("password")) {
    return;
  }
  const salts = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salts);
});

UserSchema.methods.createRefreshJWT = function () {
  return jwt.sign(
    { userId: this._id, name: this.name },
    process.env.JWT_SECRET_REFRESH,
    {
      expiresIn: "30d",
    }
  );
};

UserSchema.methods.createAccessJWT = function (extra = {}) {
  return jwt.sign(
    { userId: this._id, name: this.name, ...extra },
    process.env.JWT_SECRET_ACCESS,
    {
      expiresIn: "1h",
    }
  );
};

UserSchema.methods.checkPassword = async function (userPassword) {
  const isMatch = await bcrypt.compare(userPassword, this.password);
  return isMatch;
};

module.exports = mongoose.model("User", UserSchema);