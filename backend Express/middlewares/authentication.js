const { BadRequestError, AuthenticationError } = require("../errors/errors");
const { StatusCodes } = require("http-status-codes");
const User = require("../models/user");
const jwt = require("jsonwebtoken");

const authenticator = (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    throw new AuthenticationError("Unauthenticated");
  }
  try {
    const token = authHeader.split(" ")[1];
    const payLoad = jwt.verify(token, process.env.JWT_SECRET_ACCESS);
    req.user = {
      userId: payLoad.userId,
      name: payLoad.name,
      readonly: payLoad.readonly,
    };
    next();
  } catch (err) {
    throw new AuthenticationError("UnAuthenticated");
  }
};

module.exports = authenticator;