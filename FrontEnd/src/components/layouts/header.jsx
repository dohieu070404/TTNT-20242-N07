import React, { useEffect, useState } from "react";

import "./header.css";
import '/src/styles/icondesigns.css'

const Header = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className={`headerbar-icon ${scrolled ? "scrolled" : ""}`}>
      <div href='http://localhost:3000/' className='headerbar-icon-logopage'>
      <img src="/assets/logopage.jpg" alt="" />
      </div>

      <div className='headerbar-item'>
		<ul>
		<li>
		<a href="#" className='headerbar-btn headerbar-btn-white headerbar-btn-animated'>Giới thiệu  </a>
		</li>
		<li>
        <a href="#" className='headerbar-btn headerbar-btn-white headerbar-btn-animated'>Thuật toán </a>
		</li>
		<li>
        <a href="#" className='headerbar-btn headerbar-btn-white headerbar-btn-animated'>Tra toxic tiếng anh </a>
		</li>
		<li>
        <a href="#" className='headerbar-btn headerbar-btn-white headerbar-btn-animated'>Tra toxic Tiếng việt</a>
		</li>
		</ul>
	</div>
    <div href='#' className='headerbar-icon-user '>
        <span className=" headerbar-icon-user-color" />
         
        
      </div>
      
    </div>
  );
};

export default Header;