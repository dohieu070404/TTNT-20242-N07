import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-columns">
          <div className="footer-left">
            <h4 className="footer-title">O.O </h4>
            <p className="footer-subtitle">trông cũng oằn tà là vằn lắm </p>
            <div className="footer-socials">
              <button className="footer-icon"><i className="fab fa-twitter" /></button>
              <button className="footer-icon"><i className="fab fa-facebook-square" /></button>
              <button className="footer-icon"><i className="fab fa-dribbble" /></button>
              <button className="footer-icon"><i className="fab fa-github" /></button>
            </div>
          </div>
          <div className="footer-right">
            <div className="footer-link-group">
              <span className="footer-link-title">Tổ LINK linh tinh</span>
              <ul>
                <li><a href="#">gắn link demodemo</a></li>
                <li><a href="#">code in Github</a></li>
                <li><a href="#">giới thiệu</a></li>
              </ul>
            </div>
            
          </div>
        </div>
        <hr className="footer-separator" />
        <div className="footer-bottom">
          <p>&copy; 2025 Notus JS by bai bai ! </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;