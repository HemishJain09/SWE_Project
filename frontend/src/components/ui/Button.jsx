// Button Component
const Button = ({ onClick, children, variant = 'primary', disabled, icon: Icon }) => {
  const baseStyles = "px-6 py-3 rounded-lg font-medium transition flex items-center gap-2 justify-center";
  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-400",
    secondary: "bg-green-600 text-white hover:bg-green-700 disabled:bg-gray-400",
    outline: "border-2 border-gray-300 text-gray-700 hover:bg-gray-50"
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseStyles} ${variants[variant]} ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
    >
      {Icon && <Icon size={20} />}
      {children}
    </button>
  );
};

export default Button;