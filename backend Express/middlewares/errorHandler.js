const CustomError = require("../errors/customError");
const { StatusCodes } = require("http-status-codes");
const errorHandler = (err, req, res, next) => {
  const customError = {
    msg: err.message || "Internal server error please try again later",
    statusCode: err.statusCode || StatusCodes.INTERNAL_SERVER_ERROR,
    err: err,
  };
  if (err.code && err.code == 11000) {
    return res.status(StatusCodes.BAD_REQUEST).json({
      msg: `Duplicate value for ${Object.keys(
        err.keyValue
      )}. Please provide a unique value other than ${Object.values(
        err.keyValue
      )}`,
    });
  }
  if (err.name === "ValidationError") {
    const required = Object.keys(err.errors).join(", ");
    return res
      .status(StatusCodes.BAD_REQUEST)
      .json({ msg: `Please provide ${required} ` });
  }
  if (err.name === "CastError") {
    return res
      .status(StatusCodes.BAD_REQUEST)
      .json({ msg: `No item found with id ${err.value}` });
  }
  return res.status(customError.statusCode).json({ msg: customError.msg });
};

module.exports = errorHandler;